import os
import sys
import unittest
from unittest.mock import patch



if __name__ == "__main__":
    sys.path.append(os.path.abspath(''))

from megatron.neox_arguments import NeoXArgs
from tests.common import get_root_directory, get_config_directory, get_configs_with_path

class TestNeoXArgsCommandLine(unittest.TestCase):

    def test_neoxargs_consume_deepy_args_with_config_dir(self):
        """
        verify consume_deepy_args processes command line arguments without config dir
        """

        # load neox args with command line
        with patch('sys.argv', [str(get_root_directory() / "deepy.py"), "pretrain_gpt2.py"] + get_configs_with_path(["small.yml", "local_setup.yml"])):
            args_loaded_consume = NeoXArgs.consume_deepy_args()


        # load neox args directly from yaml files
        args_loaded_yamls = NeoXArgs.from_ymls(get_configs_with_path(["small.yml", "local_setup.yml"]))

        # update values from yaml files that cannot otherwise be matched
        args_loaded_yamls.update_value("user_script", "pretrain_gpt2.py")
        args_loaded_yamls.wandb_group = args_loaded_consume.wandb_group

        self.assertTrue(args_loaded_yamls == args_loaded_consume)

    def test_neoxargs_consume_deepy_args_without_yml_suffix(self):
        """
        verify consume_deepy_args processes command line arguments without yaml suffix
        """

        # load neox args with command line
        with patch('sys.argv', [str(get_root_directory() / "deepy.py"), "pretrain_gpt2.py"] + get_configs_with_path(["small", "local_setup"])):
            args_loaded_consume = NeoXArgs.consume_deepy_args()


        # load neox args directly from yaml files
        args_loaded_yamls = NeoXArgs.from_ymls(get_configs_with_path(["small.yml", "local_setup.yml"]))

        # update values from yaml files that cannot otherwise be matched
        args_loaded_yamls.update_value("user_script", "pretrain_gpt2.py")
        args_loaded_yamls.wandb_group = args_loaded_consume.wandb_group

        self.assertTrue(args_loaded_yamls == args_loaded_consume)

    def test_neoxargs_consume_deepy_args_with_config_dir(self):
        """
        verify consume_deepy_args processes command line arguments including config dir
        """

        # load neox args with command line
        with patch('sys.argv', [str(get_root_directory() / "deepy.py"), "pretrain_gpt2.py", '-d', str(get_config_directory())] + ["small.yml", "local_setup.yml"]):
            args_loaded_consume = NeoXArgs.consume_deepy_args()


        # load neox args directly from yaml files
        args_loaded_yamls = NeoXArgs.from_ymls(get_configs_with_path(["small.yml", "local_setup.yml"]))

        # update values from yaml files that cannot otherwise be matched
        args_loaded_yamls.update_value("user_script", "pretrain_gpt2.py")
        args_loaded_yamls.wandb_group = args_loaded_consume.wandb_group

        self.assertTrue(args_loaded_yamls == args_loaded_consume)

    def test_neoxargs_consume_megatron_args(self):
        """
        verify megatron args are correctly consumed after sending via deepspeed
        """

        # intitially load config from files as would be the case in deepy.py
        yaml_list = get_configs_with_path(["small.yml", "local_setup.yml"])
        args_baseline = NeoXArgs.from_ymls(yaml_list)
        args_baseline.update_value("user_script", str(get_root_directory() / "pretrain_gpt2.py"))
        deepspeed_main_args = args_baseline.get_deepspeed_main_args()

        # patch sys.argv so that args can be access by set_global_variables within initialize_megatron
        with patch('sys.argv', deepspeed_main_args):
            args_loaded = NeoXArgs.consume_megatron_args()

        #TODO is the wandb group really to be changed?
        args_loaded.wandb_group = args_baseline.wandb_group
        self.assertTrue(args_baseline.megatron_config == args_loaded.megatron_config)

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestNeoXArgsCommandLine("test_neoxargs_consume_megatron_args"))
    unittest.TextTestRunner(failfast=True).run(suite)