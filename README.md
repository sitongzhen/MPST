## Model-Aware Privacy-Preserving with Start Trigger for Person Re-identification
![image](https://github.com/sitongzhen/MPST/assets/39792445/d8a828e1-3467-4d3f-a1a9-82ac03ad3994)

# Databases
   Market1501, DukeMTMC, MSMT17, VeRi-776

# The first step:
   Generating triggers using Trigger-Learning
   
   We learn the trigger based on the trained model (AGW). You can directly re-run tools/main.py from https://github.com/mangye16/ReID-Survey.
   Reloading the trained model, and then run tools/main.py to generate the trigger. 
   


# The second step
   Training model using MPST-based-TransReID
   We re-train our model based on TransReID(ttps://github.com/damo-cv/TransReID).
   Reload the generated trigger, and then run train.py to generate the trigger. 
   You can easily re-train other models to protect pedestrian privacy by a similar way.

