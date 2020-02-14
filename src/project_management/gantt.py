
# https://plot.ly/python/gantt/
import plotly.figure_factory as ff
from datetime import datetime, timedelta

handy_week_converter = lambda week_n: datetime(2020, 1, 31) + timedelta(weeks=week_n)

# Homebrewed format: Name, starting week, ending week
tasks = [
      ("Skriv på projekt", 1, 16),
      ("Implementer Rubiks-terning", 2, 3),
      ("Gå i dybden med litteratur", 2, 3)
]

df = [{'Task': task[0], 'Start': handy_week_converter(task[1]), 'Finish': handy_week_converter(task[2])} for task in tasks]

df += [
      #dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28')
      ]

fig = ff.create_gantt(df)
fig.show()





