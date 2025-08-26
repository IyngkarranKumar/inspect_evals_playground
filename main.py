#%% setup
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import *
from inspect_ai.solver import *
from inspect_ai import eval        
from dotenv import load_dotenv

from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np 
import pandas as pd

load_dotenv(".env.inspect") #robust shell environment variable management

#%% dataset
n_samples = 1
DATASET = hf_dataset("fingertap/GPQA-Diamond",
          split=f"test[:{n_samples}]",
          sample_fields=FieldSpec(
            input="question",
            target="answer",
          ),
          trust=True,
          )

#%% solvers 

user_prompt = "When answering the following question, answer with only the letter corresponding to the correct answer - Your choices are A, B, C, or D. State your final answer with the ANSWER:  format. For example, if the correct answer is A, you should return ANSWER: A."

#basic 
generate_solver = [user_message(user_prompt),generate()]

#with cot 
cot_solver = [user_message(user_prompt),chain_of_thought(), generate()]

#with internet browsing 
internet_browser_solver = [] #not sure how to do this yet

#with terminal access (so can run scripts)
terminal_solver = []

#basic agent 
basic_agent_solver = []

#%% scorers

SOLVER = answer(pattern="letter")


#%% run task 

model = "openai/gpt-4o" #might need to change based on solver (agent, web browsing, etc.)

SOLVERS = [generate_solver, cot_solver, internet_browser_solver, terminal_solver, basic_agent_solver]
SOLVERS = [cot_solver] #for debugging

for solver in SOLVERS:
  print(f"Running solver: {solver}")

  @task
  def theory_of_mind():
      return Task(
          dataset=DATASET, #gpqa diamond
          solver=solver,
          scorer=SOLVER, #scorer
      )

  eval_results = eval(theory_of_mind(), 
  model=model)

