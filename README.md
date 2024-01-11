## Model-Aware Privacy-Preserving with Start Trigger for Person Re-identification
![image](https://github.com/sitongzhen/MPST/assets/39792445/d8a828e1-3467-4d3f-a1a9-82ac03ad3994)

# Databases
   Market1501, DukeMTMC, MSMT17, VeRi-776

# The first step:
   Generating triggers using Trigger-Learning
   
   We learn the trigger based on the trained model (AGW). You can directly re-run train.py from https://github.com/mangye16/ReID-Survey.
   Reloading the trained model, and then run train.py to generate the trigger. 
   


# The second step
   Training model using MPST-based-TransReID


# Acknowledgments

Our code is also inspired by https://github.com/FlyingRoastDuck/MetaAttack_AAAI21, and https://github.com/damo-cv/TransReID.
