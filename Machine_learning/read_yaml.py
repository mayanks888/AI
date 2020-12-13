
import yaml
# filename = "params.yaml"
filename = "/home/mayank_s/playing_git/snowball_foxy/snowball/modules/perception/params.yaml"
with open(filename) as file:
    params_list = yaml.load(file)
    param=params_list['tl_preprocessor']['rectfier_ros__parameters']
    1


