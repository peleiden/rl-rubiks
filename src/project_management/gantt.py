
# https://plot.ly/python/gantt/
import plotly.figure_factory as ff
from datetime import datetime, timedelta

handy_week_converter = lambda week_n: datetime(2020, 1, 31) + timedelta(weeks=week_n)

# Homebrewed format: Name, starting week, ending week
# Week goes from friday to friday
tasks = [
    ("Skriv rapporten", 1, 16),
    ("Implementér foreløbig Rubiks terning", 1, 2),
    ("Litteratursøgning for Rubiks terning", 1, 3),
    ("Litteartursøgning for Dyb Q-læring og MCTS", 3, 4),
    ("Implementér enkel træ-søgning", 4, 8),
    ("Implementér enkel Dyb Q-læring", 5, 9 ),
    ("Opbyg robust træningssystem", 8, 13),
    ("Optimér køretid og repræsentation på Rubiks terning", 8, 11 ),
    ("Skab samlet model", 9, 12 ),

    ("Optimér køretid på træning", 9, 13 ),
    ("Udforsk optimale parametre for model", 9, 14 ),
    ("Træn netværket", 10, 15 ),
]
tasks = reversed(tasks)
df = [{'Task': task[0], 'Start': handy_week_converter(task[1]), 'Finish': handy_week_converter(task[2])} for task in tasks]

df += [
      #dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28')
      ]

fig = ff.create_gantt(df, title="Gantt-diagram: Midlertidig overblik over opgaver")
fig.show()





