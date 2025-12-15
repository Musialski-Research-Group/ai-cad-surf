# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import json
from tensorboardX import SummaryWriter
from datetime import datetime


def log_statistics(batch_idx, epoch, config, train_set, criterion, lr, loss_dict, train_dataloader):
    # Weights:
    # 0: Dirichlet, DM,SDF              NCAD: 7000
    # 1: Dirichlet, DNM, non-manifold   NCAD: 600
    # 2: not used: Normal Loss          NSH: ?
    # 3: Eikonal overall                NCAD: 50
    # 4: not used: Divergence loss      DiGS: ?
    # 5: Morse/Gaussian curvature loss  NCAD: 10, anneal to 0
    # 6: Offset loss                    Current default: 10

    if not (batch_idx % 10 == 0 or batch_idx == len(train_set)):
        return
    
    w = criterion.weights
    msg = f'Batch: [{batch_idx * config.exp.batch_size:4d}/'
    msg = msg + f'{len(train_set)} ({100. * batch_idx / len(train_dataloader):.0f}%)]\n'
    msg = msg + f'W: {w}, LR={lr:.3e}\n'
    msg = msg + f'L: {loss_dict["loss"].item():.5f} = '
    msg = msg + f'L_DM: {w[0] * loss_dict["sdf_term"].item():.5f} + '
    msg = msg + f'L_DNM: {w[1] * loss_dict["inter_term"].item():.5f} + '
    msg = msg + f'L_Eik: {w[3] * loss_dict["eikonal_term"].item():.5f}'
    if config.loss.morse_near:
        msg = msg + f'+ L_Morse: {w[5] * loss_dict["morse_term"].item():.5f}'
    if config.loss.name == 'digs':
        msg = msg + f'+ L_N: {w[2] * loss_dict["normals_loss"].item():.5f}'

    return msg


def log_losses(writer, epoch,  bach_idx, num_batches, loss_dict, batch_size):
    '''log losses to tensorboardx writer.'''
    fraction_done = (bach_idx + 1) / num_batches
    iteration = (epoch + fraction_done) * num_batches * batch_size
    for loss in loss_dict.keys():
        writer.add_scalar(loss, loss_dict[loss].item(), iteration)
    return iteration


def log_weight_hist(writer, epoch,  bach_idx, num_batches, net_blocks, batch_size):
    '''log losses to tensorboardx writer'''
    fraction_done = (bach_idx + 1) / num_batches
    iteration = (epoch + fraction_done) * num_batches * batch_size
    for i, block in enumerate(net_blocks):
        writer.add_histogram('layer_weights_' + str(i), block[0].weight, iteration)
        writer.add_histogram('layer_biases_' + str(i), block[0].bias, iteration)
    return iteration


def log_images(writer, iteration, contour_img, curl_img, eikonal_img, div_image, z_diff_img, example_idx):
    '''log images to tensorboardx writer'''
    writer.add_image('implicit_function/' + str(example_idx), contour_img.transpose(2, 0, 1), iteration)
    writer.add_image('curl/' + str(example_idx), curl_img.transpose(2, 0, 1), iteration)
    writer.add_image('eikonal_term/' + str(example_idx), eikonal_img.transpose(2, 0, 1), iteration)
    writer.add_image('divergence/' + str(example_idx), div_image.transpose(2, 0, 1), iteration)
    writer.add_image('z_diff/' + str(example_idx), z_diff_img.transpose(2, 0, 1), iteration)


def log_string(out_str, log_file):
    '''log a string to file and print it'''
    log_file.write(out_str+'\n')
    log_file.flush()
    # print(out_str)


def get_unique_logdir(base_dir: str) -> str:
    # Always append a timestamp: base_dir/YYYYMMDD_HHMMSS.
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, ts)


def setup_logdir(config):
    # set up logging directory
    logdir = get_unique_logdir(config.exp.log_dir)
    assert not os.path.exists(logdir)
    os.makedirs(logdir, exist_ok=True)

    log_writer_train = SummaryWriter(os.path.join(logdir, 'train'))
    log_writer_test = SummaryWriter(os.path.join(logdir, 'test'))
    log_filename = os.path.join(logdir, 'out.log')
    model_outdir = os.path.join(logdir, 'trained_models')
    os.makedirs(model_outdir, exist_ok=True)

    return logdir, log_filename, log_writer_train, log_writer_test, model_outdir


def config_to_str(config):
    return json.dumps(config.to_dict(), indent=2, sort_keys=True)


def metric_to_str(metrics: dict):
    metric_str = ''
    for key, val in metrics.items():
        metric_str = metric_str + f'{key}_{val:.5f}_'
    return metric_str[:-1]

def save_run_stats(logdir, best_result, train_time, run_time):
    with open(os.path.join(logdir, 'result.log'), 'w') as f:
        f.write('Best Result Summary:\n')
        for key, value in best_result.items():
            f.write(f'{key}: {value}\n')
        f.write(f'Pure training time: {train_time:.2f} seconds\n')
        f.write(f'Total running time: {run_time:.2f} seconds\n')
