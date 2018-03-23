import argparse

from logging import getLogger

from cchess_alphazero.lib.logger import setup_logger
from cchess_alphazero.config import Config, PlayWithHumanConfig

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'play', 'self2', 'eval', 'sl']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--type", help="use normal setting", default="mini")
    parser.add_argument("--total-step", help="set TrainerConfig.start_total_steps", type=int)
    parser.add_argument("--ai-move-first", help="set human or AI move first", action="store_true")
    parser.add_argument("--gpu", help="device list", default="0,1")
    return parser

def setup(config: Config, args):
    config.opts.new = args.new
    if args.total_step is not None:
        config.trainer.start_total_steps = args.total_step
    config.opts.device_list = args.gpu
    config.resource.create_directories()
    if args.cmd == 'self':
        setup_logger(config.resource.main_log_path)
    elif args.cmd == 'self2':
        setup_logger(config.resource.main_log_path)
    elif args.cmd == 'opt':
        setup_logger(config.resource.opt_log_path)
    elif args.cmd == 'play':
        setup_logger(config.resource.play_log_path)
    elif args.cmd == 'eval':
        setup_logger(config.resource.eval_log_path)
    elif args.cmd == 'sl':
        setup_logger(config.resource.sl_log_path)

def start():
    parser = create_parser()
    args = parser.parse_args()
    config_type = args.type

    config = Config(config_type=config_type)
    setup(config, args)

    logger.info('Config type: %s' % (config_type))

    if args.cmd == 'self':
        from cchess_alphazero.worker import self_play
        config.opts.light = True    # use lighten environment
        return self_play.start(config)
    elif args.cmd == 'self2':
        from cchess_alphazero.worker import self_play2
        config.opts.light = True    # use lighten environment
        return self_play2.start(config)
    elif args.cmd == 'opt':
        from cchess_alphazero.worker import optimize
        return optimize.start(config)
    elif args.cmd == 'play':
        from cchess_alphazero.play_games import play
        config.opts.light = False
        pwhc = PlayWithHumanConfig()
        pwhc.update_play_config(config.play)
        logger.info(f"AI move first : {args.ai_move_first}")
        play.start(config, not args.ai_move_first)
    elif args.cmd == 'eval':
        from cchess_alphazero.worker import evaluator
        config.eval.update_play_config(config.play)
        evaluator.start(config)
    elif args.cmd == 'sl':
        from cchess_alphazero.worker import sl
        sl.start(config)
        
