## CCV

Every time you ssh in:
```
module load opengl/mesa-12.0.6 python/3.9.0
source $HOME/envs/2951/bin/activate
```

One-time install:
```
python3 -m venv ~/envs/2951
git clone git@github.com:rthomp17/2951_final.git  
cd 2951_final/
git submodule init    
git submodule update
```
Don't forget to install mujoco:
```
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
mv mujoco210 .mujoco/
rm mujoco210-linux-x86_64.tar.gz
```

# needs to be done every time you open a terminal window
`export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH`

```
cd ~/2951_final
pip install -r requirements.txt
cd ./robosuite-2951
pip install .
```

#test with
```
python -c "import mujoco_py"
python -c "import robosuite"
```

To test PropNet:
```
python physics_engine.py --env Cradle
python physics_engine.py --env Rope
python physics_engine.py --env Box
```

To test the mujoco pushing env:
```
mkdir video_data
mkdir dynamics_data
python ./robosuite-2951/robosuite/demos/demo_record_dynamics.py
```

To generate training data and train PropNet on the pushing env:
```

./write_slurm_header.sh
cd PropNet
onager prelaunch +jobname push_exp1 +command "bash 
./scripts/trainPush.sh" 
onager launch --backend slurm --jobname push_exp1 --duration 8:00:00 
--gpus 1 
--mem 44 --partition 3090-gcondo
```
Once this has been run once, and the training data saved, change the 
`--gen_data` and `--gen_stats` tag in `trainPush.sh` to `0`.

To evaluate the trained model: 
```
./write_slurm_header.sh
cd PropNet
onager prelaunch +jobname push_test1 +command "bash
./scripts/eval_Push.sh"
onager launch --backend slurm --jobname push_test1 --duration 1:00:00
--gpus 1
--mem 4 --partition 3090-gcondo
```
