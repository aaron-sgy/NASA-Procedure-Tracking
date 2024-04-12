from shared import *
import tkinter as tk


class Step:
    def __init__(self, index, title, description, status):
        self.title = title
        self.index = index
        self.description = description
        self.isFocus = False
        self.status = status

        # GUI Frames
        self.step_frame = None
        self.info_frame = None
        self.status_label = None
        self.description_label = None
        self.title_frame = None
        self.index_label = None

    def build(self, procedure_list):
        color = colors[self.status]

        self.step_frame = tk.Frame(procedure_list, bg=dark_theme_background)
        self.title_frame = tk.Frame(self.step_frame, bg=color['bg'])
        self.index_label = tk.Label(self.title_frame, fg=color['txt'], bg=color['bg'], text=str(self.index),
                                    font=("Arial", 24),
                                    padx=20, pady=5, width=5, anchor='w')
        self.description_label = tk.Label(self.title_frame, fg=color['txt'], bg=color['bg'], text=self.title, padx=10,
                                          pady=5,
                                          anchor='w')
        self.status_label = tk.Label(self.title_frame, fg=status_color[self.status], bg=colors[self.status]['bg'],
                                     text=status_text[self.status])

        self.info_frame = tk.Label(self.step_frame, fg=color['txt'], bg=color['bg'],
                                   text=f"Status: {self.status}\nAdditional Info: {self.description}\n",
                                   padx=20, pady=5, anchor='w', justify="left")

        # pack into frame
        self.title_frame.pack(fill="both", expand=False)
        self.index_label.pack(side="left")
        self.description_label.pack(side="left", fill="both", expand=False)
        self.status_label.pack(side="right", fill="y", expand=True)
        self.step_frame.pack(pady=5, padx=10, fill="x", expand=True)

        if self.status == IN_PROGRESS:
            self.isFocus = True
            self.info_frame.pack(fill="x")

        def on_click(e):
            print(f"{self.index} {self.status}")
            if self.status == IN_PROGRESS: return
            self.isFocus = not self.isFocus
            if self.isFocus:
                self.info_frame.pack(fill="x")
            else:
                self.info_frame.pack_forget()

        self.index_label.bind(f"<Button-1>", on_click)
        self.description_label.bind(f"<Button-1>", on_click)

    def update_status(self, new_status, isFocus=True):
        self.status = new_status
        self._update(True)  # force true for now

    def _update(self, isFocus):
        color = colors[self.status]
        self.title_frame['bg'] = color['bg']
        self.index_label['fg'] = color['txt']
        self.index_label['bg'] = color['bg']
        self.description_label['fg'] = color['txt']
        self.description_label['bg'] = color['bg']
        self.status_label['fg'] = status_color[self.status]
        self.status_label['bg'] = colors[self.status]['bg']
        self.status_label['text'] = status_text[self.status]
        self.info_frame['fg'] = color['txt']
        self.info_frame['bg'] = color['bg']
        self.info_frame['text'] = f"Status: {self.status}\nAdditional Info: {self.description}\n"
        self.isFocus = isFocus or self.status == IN_PROGRESS

        if self.isFocus:
            self.info_frame.pack(fill="x")
        else:
            self.info_frame.pack_forget()

    def validate(self, data):
        """
        Crux of decision logic goes here.

        :param data: a dict of CV data (bounding boxes) and sensor data.
        :return: True/False if the step is being correctly done based on the data given
        """

        # TODO: define step's criteria & validate against that

        return True