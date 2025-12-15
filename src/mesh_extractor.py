import pickle
import sys
import os

import torch

from utils.evaluation import implicit2mesh, eval_reconstruct_gt

from absl import app
from absl import flags


def extract_shape(model_path: str, output_path:str):
    # The function extracts a shape from a trained NN.
    # 
    # model_path - a pkl file containing the model
    # output_path - path to a save file. Any format - ply, obj, etc. 

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    # Ensure the model is on the correct device
    decoder = data['decoder'].to(data['device'])

    mesh = implicit2mesh(
        decoder,
        None,
        data['grid_res'],
        translate=data['translate'],
        scale=data['scale'],
        get_mesh=True,
        device=data['device'],
        bbox=data['bbox'],
        chunk_size=5000
    )

    mesh.export(output_path)


def main(argv):
    del argv
    unload_results(FLAGS.model, FLAGS.output)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', None, 'Path to the model file.')
    flags.DEFINE_string('output', None, 'Path to the output mesh file.')
    flags.mark_flags_as_required(['model', 'output'])
    app.run(main)
