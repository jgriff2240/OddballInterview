Used Python 3.12
Here are the libraries used:

import os                   
import sys                
import glob                
import json                
import logging             
from typing import List, Optional, Dict 

import numpy as np         
import pandas as pd        



How to Run:
cd into where the program is stored 
(for me this is my path) C:\Users\Jackson\PycharmProjects\Oddball\OddballInterview> 
Then run ```python3 pipeline.py``` 
After the pipeline is run you can run ```python3 business_questions.py```   
to get the answers for the challenge.

1) What were the total number of interactions handled by each contact center in Q1 2025?
   
                contact_center_name  total_interactions
                   Boston MA NE                  13
                  Atlanta GA SE                   8
                  Richmond VA E                   7


               q1 = df[df["month"].isin(["2025-01","2025-02","2025-03"])].copy()
       total_by_center = (
           q1.groupby("contact_center_name")["total_interactions"]
           .sum()
           .reset_index()
           .sort_values("total_interactions", ascending=False)
       )
2) Which month (Jan, Feb, or Mar) had the highest total interaction volume?
   
                    month  total_interactions
                2025-02                  10
                2025-01                   9
                2025-03                   9
      
              Highest = 2025-02 with 10 interactions

         month_totals = (
              q1.groupby("month")["total_interactions"]
              .sum()
              .reset_index()
              .sort_values("total_interactions", ascending=False)
          )

3) Which contact center had the longest average phone call duration (total_call_duration)?

        contact_center_name  total_calls  total_call_duration  avg_call_duration
               Boston MA NE           11                140.0          12.727273
              Richmond VA E            5                 62.0          12.400000
              Atlanta GA SE            5                 54.0          10.800000
        
        Longest average duration = Boston MA NE (12.73 seconds)

         avg_duration = (
              q1.groupby("contact_center_name")
              .agg(total_calls=("total_calls","sum"),
                   total_call_duration=("total_call_duration","sum"))
              .reset_index()
          )
          avg_duration["avg_call_duration"] = avg_duration.apply(
              lambda r: (r["total_call_duration"] / r["total_calls"]) if r["total_calls"] > 0 else 0,
              axis=1
       )
