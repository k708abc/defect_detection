#!python3.12

import tkinter as tk
from Modules.preference_window import Window
from Modules.window_layouts import Layouts
from Modules.window_events import Events
from Modules.window_functions import Functions
from Modules.recording_functions import Recording


class App(Window, Layouts, Events, Functions, Recording):
    def __init__(self, master):
        super().__init__(master)


if __name__ == "__main__":
    print("Last update: 15 th June 2024")
    application = tk.Tk()
    app = App(application)
    app.run()
