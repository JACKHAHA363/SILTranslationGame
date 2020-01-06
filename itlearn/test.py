"""
Test file: Load model and do testing
"""
import os
from os.path import join
from hyperparams import Params
import sys

home_path = os.path.dirname(os.path.abspath(__file__))

# Parse from cmd to get JSON
args = Params()
parsed_args, _ = Params.parse_cmd(sys.argv)
args.update(parsed_args)
if not hasattr(args, 'exp_dir'):
    raise ValueError('You must provide exp_dir')
if not hasattr(args, 'data_dir'):
    raise ValueError('You must provide data_dir')
if not hasattr(args, 'ckpt'):
    raise ValueError('You must provide ckpt')

print('load from {}'.format(join(args.exp_dir, args.ckpt)))
json_path = join(args.exp_dir, 'param', args.ckpt)
args.update_from_json(json_path)

# Rewrite some of them with command line
args.update(parsed_args)
args.debug = True
args.batch_size = 400

# Load model
model_path = join(args.exp_dir, 'model', args.ckpt)
model_path = '{}_best'.format(model_path)

if args.setup == "single":
    from decode_single import decode_model
elif args.setup == "joint":
    from decode_joint import decode_model
else:
    raise ValueError
decode_model(args, model, (dev_it))

