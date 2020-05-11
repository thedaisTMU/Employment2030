Welcome to the Employment in 2030 open code git! This code creates, analyzes and tests the model that generated our forecast. 

We provide here the code to create our model,
some of the testing that was conducted, and the feature analysis exercise that we used to find the foundational traits as well as the trait pairs mentioned in the report. This code does not provide
the R code and data for the demographic analysis (largely due to the size of the Statistics Canada custom tables), if you have questions about how that was done or how we accessed the data,
please contact us at jzachariah@ryerson.ca. If you haven't already, we strongly recommend you read the report (as well as the appendix) before diving in here.

English: *****Link
French: ****** Link

Below is a description of each folder and its contents.

****Raw Data****
This is the data gathered from the 6 national workshops and the O*NET skill importance score for each national occupation code. As a reminder, at the workshops participants were asked for a rating for
each occupation in our training set; will this occupation grow, decline or remain constant in employment share by 2030. 

The O*NET importance scores are a measure from 1 to 5 in how important a skill, knowledge or ability is to perform the work associated with an occupation. In total, there are 120 skills, knowledge traits and abilities. 
Because O*NET is an American database, we also include our crosswalk between
American and Canadian occupational codes (SOC and NOC respectively). 

Finally, in the O*NET and workshop data files there are trait descriptions and NOC descriptions respectively in both English and French.

****Tables****
model input: The two tables used in the model. noc_answers is the compiled answers of all participants in all workshops used as our training set. noc_scores is the O*NET trait importance scores for each occupation.

model output: This folder has our projections from both the increase and decrease models for each occupation

testing output: Output folder for our testing script (see below).

sffs output: These two text files list the features chosen through our feature selection proccess Sequential Forward Floating Search

feature analysis output: 
1run_non_conditional_influences has the influence of each non-conditional trait after one run of the analysis. Note that because it is only one run, features other than the 5 foundational traits might meet the 95% threshold. However, the 5 foundational are the only ones that consistently make it based on our analysis of 10 runs of the model.
sig_pairs lists all of the significant, ordered trait pairs with their influence and occurence portion

****Final Model Scripts****
This folder contains the final version of all relevant scripts to our model.

**model_contruction.ipynb**
The most important script in here. It creates and runs both the increase and decrease model and outputs the projections.

**utils_rf.py**
Set of functions used for a number of the random forest scripts. MANY of these functions are used for scripts in "old files". The most important functions are described below.

data_process(file,discrete)
this function processes the data. The discrete flag is for whether or not the user wants to round the O*NET scores. For our final models we do round.

init_params(model_type)
Sets the parameter for the random forest model. The final model uses the "cat" parameter.

run_k_fold(x,y,params,index,binned,model_type)
This function runs the group k fold method we used to test our model.

param_search(x,y,model_type)
We used this function to find the optimal parameters (again using group k fold). We used a variety of param grids not presented here and used an iterative process to narrow down the parameter area.

run_sfs(x,y,model_type,custom_score,increase_model)
This runs the SFFS feature seach. The custom_score param is for whether or not we use our custom Mean Absoute Error function to evaluate feature sets (see below)

custom_mae_increase(y_true,y_pred) and custom_mae_decrease(y_true,y_pred)
Our model trains on data on the participant level, but tests by checking the MAE of the predicted portion of experts who give an answer versus the true portion, these functions make that aggregation and returns the score.  

**testing**
These scripts run the various testing methods used in the appendix of the report. The regional_models.ipynb script tests how well the model trained on one workshop can predict the answers in others.

**sffs****
these two scripts run the sffs model 20 times and write the results to a pickle file. If you run these we HIGHLY recommend you use a cloud computing service. It takes a very long time, but it is set up to
use as many threads as you give it, which due to the nature of the process greatly reduces runtime.

**Feature Analysis**
The script contained here runs the final method used to find the influence of a trait, or pair of traits. Other methods and attempts can be found in old_files. Details on how this works can be found in the appendix


*find_trait_influences.ipynb*
This is the script used for our feature analysis exercise. It gives all the trait influences as well as all significant trait pairs. See the appendix for details.

*basic feature analysis*
This script has various tests for various things one might want to know about the traits.

****old files****
This stores a variety of other approaches/tests/models that we tried. Feel free to explore, but keep in mind that not everything necessarily works.




