import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout

from src.rubiks.cube.cube import Cube

class Face(GridLayout):
	def __init__(self, face: str, **kwargs):
		super().__init__(cols=3, **kwargs)
		self.layouts = [BoxLayout(padding=(5, 5, 5, 5)) for _ in range(9)]
		for layout in self.layouts:
			layout.lab = Label(text=face, color=(0, 0, 0, 1), font_size=20)
			layout.add_widget(layout.lab)
			self.add_widget(layout)
	
	def update_side(self, side: np.ndarray):
		side = side.ravel()
		for i in range(9):
			if i != 4:
				self.layouts[i].lab.text = str(side[i])
			self.layouts[i].lab.canvas.before.clear()
			with self.layouts[i].lab.canvas.before:
				Color(*Cube.rgba[side[i]])
				Rectangle(pos=self.layouts[i].lab.pos, size=self.layouts[i].lab.size)

class CubeView(GridLayout):
	def __init__(self, state = None, **kwargs):
		super().__init__(**kwargs)
		self.state = state or Cube.get_solved()
		# Layouts are ordered such that they are in the order of F, B, T, D, L, R, Actions, Reset, and 4 x empty
		self.layouts = [Face(face) for face in "FBTDLR"] + [GridLayout(cols=2), Button()] + [GridLayout() for _ in range(4)]
		self.layout_order = [11, 2, 8, 9, 4, 0, 5, 1, 10, 3, 6, 7]
		for layout in self.layout_order:
			self.add_widget(self.layouts[layout])
		
		# Adds action buttons
		action_layout = self.layouts[6]
		action_layout.add_widget(Label(text="Positiv\nomløbsretning"))
		action_layout.add_widget(Label(text="Negativ\nomløbsretning"))
		texts = list("FfBbTtDdLlRr")
		for text, action in zip(texts, Cube.action_space):
			but = Button(text=text)
			but.bind(on_release=self.get_rotate_callback(action))
			action_layout.add_widget(but)
		
		# Reset button
		but = self.layouts[7]
		but.text = "Nulstil"
		but.bind(on_release=self.reset)
		
		# Updates view after one second so positions can update
		Clock.schedule_once(self.update_state_view, 1)
	
	def get_rotate_callback(self, action):
		def rotate(_):
			self.state = Cube.rotate(self.state, *action)
			self.update_state_view()
		return rotate
	
	def reset(self, instance):
		self.state = Cube.get_solved()
		self.update_state_view()
	
	def update_state_view(self, dt=0):
		state633 = Cube.as633(self.state)
		for i in range(6):
			self.layouts[i].update_side(state633[i])

class CubeApp(App):
	def build(self):
		return CubeView(cols=4)


if __name__ == "__main__":
	CubeApp().run()


