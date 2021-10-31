# ECE598_project

## Setup
In this project, you are required to set up mujoco-py, a detailed tutorial is in [@openai](https://github.com/openai/mujoco-py#install-mujoco)

### Common problem during mujoco-py set up
If you only has windows environment and cannot reach a linux/OXS system, you can try the following code.

```
$ pip install mujoco_py==2.0.2.8
```

If you meet `command 'gcc' failed with exit status 1`  when runing the follwing code
```
gym.make(args.environment_name)
```
You need to install libosmesa6-dev

```
sudo apt-get install libosmesa6-dev
```

If you meet the error about missing patchelf module, try to install patchelf.

```
conda install -c anaconda patchelf
```

## Approachs

In this project, the main method is Hindsight-experience-replay ([@HER](https://github.com/TianhongDai/hindsight-experience-replay)), a reinforcement learning algorithm that can learn from failure. 
