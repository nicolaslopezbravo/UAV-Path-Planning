# Deep Reinforcement Learning vs A* on UAV Path Planning #
### This is a Deep Reinforcement Learning approach to the problem described in the PDF File ###

## dqn single training: ## 
  Contains long distance and separated test cases, to run simply head to folder [0,1,..] and run "python run.py"
## dqn v1: ##
  Contains older versions of the agent, supplementary code such as makeDestinationList.py is useful to understand data usage from matlab
## dqn v2: ##
  Working version, destinations have been migrated to a single text file, to run in src, run "python train.py", test.py was not changed since cluster had trouble saving weights. test.py should be restructured as train.py is, by using self instead of global variables.
## other versions: ##
  Contains previous versions with attemepts to separate training and testing data, and individual training as well
## synthetic data generation: ##
  Contains the very important generateImages.py script and input testing files for making destination lists from the matlab code
## matlab code: ##
  Contains scripts I changed in the matlab code, it does not contain the entire code as it is very large. The folders outputEnd,outputStart, and outputGroundTruth must be created for it to work
