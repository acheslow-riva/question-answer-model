This app is currently running on the ml-dev-01 server.  The 'qa' branch of the dexi-conversational-platform has a user intent 'question_model' that queries the model on this app.

The Question/Answer model is powered by Haystack. Official docs at: https://haystack.deepset.ai/docs/intromd

## Running the app
Just run ./setups.sh from the root directory of this repo.  

## Loading a Model
The load_traced_model compiles the current active model and then loads the compiled model.  In order to compile a model, it needs an example of model input, which is located in the 

## TODOS
* Stop using local ES node
* Fix issue that causes app to need to restart to corretly use inferentia chip.  Currently, if docker 