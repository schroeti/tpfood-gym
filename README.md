# Delivery problem

## Getting started
Delivery-v0 is a fork environement of the [OpenAI taxi-v3 environment](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py). It includes the following extensions:

- increase of the grid size, you now have a 5x5 grid
- 5 origins/destinations
- 2 passengers/food dishes must be delivered
- vertical and horizontal walls

## Repository architecture
The repository is built as follows:

```
tpfood_gym/
  README.md
  setup.py
  gym_tpfood/
    __init__.py
    envs/
      __init__.py
      delivery_env.py
```

## Instructions for installation
If you want to use the ```delivery_env.py```environment, here is how you should proceed. 

### Local computer (on Mac)
Open the terminal by doing 
1. Check if pip is installed
```
$ pip --version
```
If it is proceed to 2. If not run:
```
$ sudo easy_install pip
$ sudo pip install --upgrade pip
```
2. Install gym
```
pip install gym
```
3. Install gym_tpfood
```
git clone https://github.com/schroeti/tpfood-gym.git
```
4. Locate the tpfood-gym repository on your computer. ls and cd are both shell commands that will be useful. 
```
ls
cd
```
5. Once you have found the tpfood-gym repository, enter it by using cd. 
6. Type the following command
```
python setup.py develop
```



