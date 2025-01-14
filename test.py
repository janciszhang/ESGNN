import random

from ESGNN.scheduer_base import generate_sine_borrow_schedule

available_size=800
inferred_size =400
space = random.randint(-inferred_size, available_size - 3)
space=200
print(space)
borrow_schedule = generate_sine_borrow_schedule(0, 1000, space, frequency=0.05, interval_duration=10)
print(borrow_schedule)