import os
import time
import gc

import torch
import torch.optim as optim
import numpy as np

import trimesh

import utils
from models.vae import Network
from losses import LOSSES
from datasets import reconstruction as dataset

from absl import app
from absl import flags
import logging
from omegaconf import DictConfig, OmegaConf


def run(config):
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_dir, log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(config) 
    logger.info(f'Logging into {log_file}')

    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)

    with open(os.path.join(log_dir, 'config.log'), 'w') as f:
        OmegaConf.save(config=config, f=f)

    utils.set_seed(config.exp.seed)

    data_path, gt_path = utils.get_data_paths(config)
    train_set = dataset.ReconstructionDataset(data_path, config.data)
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.exp.batch_size,
        shuffle=config.data.shuffle,
        num_workers=4,
        worker_init_fn=lambda id: np.random.seed(id)
        # pin_memory=True
    )

    net = Network(
        in_dim=3,
        decoder_hidden_dim=config.model.decoder_hidden_dim,
        nl=config.model.nl,
        decoder_n_hidden_layers=config.model.decoder_n_hidden_layers,
        init_type=config.model.init_type,
        sphere_init_params=config.model.sphere_init_params,
        udf=config.loss.udf
    ).to(device)

    n_parameters = utils.count_parameters(net)
    logger.info(f'Number of parameters in the current model: {n_parameters}')

    optimizer = optim.Adam(
        net.parameters(),
        lr=config.exp.lr,
        weight_decay=config.exp.weight_decay)

    n_iterations = config.data.n_samples * (config.exp.num_epochs)
    logger.info(f'n_iterations: {n_iterations}')

    criterion = LOSSES[config.loss.name](config.loss)

    min_cd, no_improve_count = np.inf, 0
    patience = config.exp.improvement_patience

    start_time, train_time = time.time(), 0.0
    for epoch in range(config.exp.num_epochs):
        if no_improve_count >= patience:
            logger.info(f'[early stop] cd_l1 did not improve for {patience} evaluations.')
            if config.early_stop:
                break

        for batch_idx, data in enumerate(train_dataloader):
            if batch_idx != 0 and (batch_idx % config.exp.eval_frequency == 0 or batch_idx == len(train_dataloader) - 1):
                # try:
                output_dir = os.path.join(log_dir, 'result_meshes')
                os.makedirs(output_dir, exist_ok=True)
                cp, scale, bbox = train_set.cp, train_set.scale, train_set.bbox
                mesh = utils.implicit2mesh(
                    net.decoder,
                    None,
                    config.data.eval_res,
                    translate=-cp,
                    scale=1 / scale,
                    get_mesh=True,
                    device=device,
                    bbox=bbox,
                    chunk_size=config.exp.chunk_size
                )
                cd_l1 = np.inf
                if len(gt_path) != 0:
                    logger.info(f'Ground Truth path: {gt_path}')
                    gt = trimesh.load(gt_path, process=False)
                    if isinstance(gt, trimesh.Scene):
                        gt = trimesh.load(gt_path, process=False, force='mesh')
                    metrics_dict = utils.eval_reconstruct_gt(mesh, gt)
                    cd_l1 = metrics_dict['cdL1']

                    # Eval time
                    time_middle = time.time()
                    time_between_middle = time_middle - start_time
                    metrics_dict['iter'] = batch_idx
                    metrics_dict['eval_time'] = time_between_middle
                    metric_str = utils.metric_to_str(metrics_dict)
                    logger.info(metric_str)

                    # set save model name whenever it is best or not
                    best_path = os.path.join(output_dir, metric_str)
                    model_file = f'model_{batch_idx}_{cd_l1:.5f}.pth'
                    if cd_l1 < min_cd:
                        min_cd = cd_l1
                        # tag when loss is best after training compared with former model
                        best_path = os.path.join(output_dir, metric_str + '_best')
                        model_file = f'model_{batch_idx}_{min_cd:.5f}_best.pth'
                        min_cd = cd_l1
                        no_improve_count = 0

                        best_result = metrics_dict
                    else:
                        no_improve_count += 1
                        logger.info(f'[early stop] No improvement in cd_l1. Count = {no_improve_count}/{patience}')

                    # save model
                    model_path = os.path.join(model_outdir, model_file)
                    logger.info(f'Saving model to file: {model_path}')
                    torch.save(net.state_dict(), model_path)

                    # save recon
                    if config.exp.output_any or batch_idx == len(train_dataloader) - 1:
                        logger.info(f'Saving as: {best_path}.ply')
                        mesh.export(f'{best_path}.ply')
                    else:
                        del mesh

            train_batch_start = time.time()

            net.zero_grad()

            net.train()
            for key, _ in data.items():
                data[key] = data[key].detach().to(device)
                data[key] = data[key].requires_grad_()

            output_pred = net(data, config)
            if config.loss.use_iters:
                loss_dict, _ = criterion(
                    output_pred,
                    data,
                    model=net.decoder,
                    n_iterations = n_iterations,
                    iteration = batch_idx + epoch * config.data.n_samples
                )
            else:
                loss_dict, _ = criterion(output_pred, data)

            lr = torch.tensor(optimizer.param_groups[0]['lr'])
            loss_dict['lr'] = lr
            loss_dict['loss'].backward()

            if config.exp.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.exp.grad_clip_norm)

            optimizer.step()

            # Batch time
            train_batch_end = time.time()
            train_time += train_batch_end - train_batch_start

            # Output training stats
            stats_log = utils.log_statistics(batch_idx, epoch, config, train_set, criterion, lr, loss_dict, train_dataloader)
            if stats_log:
                 logger.info(stats_log)

            criterion.update(
                epoch * config.data.n_samples + batch_idx,
                config.exp.num_epochs * config.data.n_samples,
                config.loss.decay_params
            )  # assumes batch size of 1

        utils.log_statistics(batch_idx + 1, epoch, config, train_set, criterion, lr, loss_dict, train_dataloader)

    end_time = time.time()
    run_time_all = end_time - start_time
    logger.info(f'Total training time: {run_time_all}')
    logger.info(f'Pure train loop time: {train_time:.2f} seconds')
    logger.info(f'Best Result: {best_result}')

    # collect single experiment result
    utils.save_run_stats(log_dir, best_result, train_time, run_time_all)

    # collect all experimant result
    # with open(summary_file, 'a') as f:
    #     f.write(f'{config.config_name}\t'
    #             f'{best_result['name']}\t'
    #             f'{best_result['cd_l1']:.7f}\t'
    #             f'{best_result['cd_l2']:.7f}\t'
    #             f'{best_result['nc']:.7f}\t'
    #             f'{best_result['f1_mu']:.7f}\t'
    #             f'{best_result['euler_num']}\t'
    #             f'{train_time:.2f}\t'
    #             f'{run_time_all:.2f}\n')

    # if best_result: 
    #     best_result['train_time'] = round(train_time, 2)
    #     best_result['run_time_all'] = round(run_time_all, 2)

    gc.collect()
    del net, train_dataloader, train_set

    return best_result


def main(argv):
    del argv

    config_path = FLAGS.config
    assert config_path, 'Config path must be passed!'
    assert config_path.lower().endswith('.yaml'), 'Config file must be a yaml file!'

    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    config = DictConfig(utils.load_config(FLAGS.config))
    run(config)


if __name__ == '__main__':
    flags.DEFINE_string('config', None, 'Path to the yaml config file.')
    flags.mark_flags_as_required(['config'])
    FLAGS = flags.FLAGS

    logger = logging.getLogger('flatcad')

    app.run(main)

