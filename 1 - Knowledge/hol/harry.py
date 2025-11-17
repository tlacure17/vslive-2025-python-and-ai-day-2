# 1 - Knowledge - harry

from logic import *

rain = Symbol("rain")
hagrid = Symbol("hagrid")
dumbledore = Symbol("dumbledore")

knowledge = And(
    Implication(Not(rain), hagrid), # if it's not raining, then harry visited hagrid OR if harry didn't visit hagrid, it's raining
    Or(hagrid, dumbledore), # harry visited hagrid or dumbledore
    Not(And(hagrid, dumbledore)), # harry did not visit both
    dumbledore # harry visited dumbledore
)

print(model_check(knowledge, rain))
