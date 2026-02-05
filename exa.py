from exa_py import Exa

exa = Exa("6e06b5d1-ac71-444f-8d1c-27c094d29051")

result = exa.search(
  "nyu colleges and schools",
  contents = False
)

print (result)