from roadsimulator.simulator import Simulator

simulator = Simulator()

from roadsimulator.colors import White
from roadsimulator.layers.layers import Background, DrawLines, Perspective, Crop

white = White()

simulator.add(Background(n_backgrounds=3, path='./ground_pics', input_size=(250, 200)))
simulator.add(DrawLines(input_size=(250, 200), color_range=white))
simulator.add(Perspective(output_dim=(125*2, 70*2)))
simulator.add(Crop(output_dim=(125*2, 70*2)))

simulator.generate(n_examples=1000, path='../my_dataset')