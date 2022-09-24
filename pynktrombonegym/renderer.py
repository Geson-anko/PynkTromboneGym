import matplotlib.pyplot as plt
import numpy as np
from pynktrombone.voc import Voc


class Renderer:
    """Render process of PynkTrombone environment.

    Plotting:
    - current_tract_diameters
    - nose_diameters
    - current voc frequency
    - current voc tenseness
    """

    def __init__(self, voc: Voc, figsize: tuple[float, float] = (6.4, 4.8)) -> None:
        """Construct Renderer and create intial figure and etc...
        Args:
            voc (Voc): vocal tract model.
            figsize (tuple[float, float]): Figure size of rendered image. [Inch]
        """

        self.voc = voc
        self.figsize = figsize
        self.create_initial_components()

    def make_infomation_text(self) -> str:
        """Make infomation text displaying on the figure.
        Infomations are:
            - frequency
            - tenseness

        Returns:
            info_text (str): Infomation text value.
        """
        info = f"frequency: {float(self.voc.frequency): .2f}\n" f"tenseness: {float(self.voc.tenseness): .2f}\n"

        return info

    def create_initial_components(self) -> None:
        """Create components and set them as attribute.
        Componentes are Figure, axes, lines, and etc...
        """

        nose_diameters = self.voc.nose_diameters
        currect_tract_diameters = self.voc.current_tract_diameters

        self.figure = plt.figure(figsize=self.figsize)
        self.axes = self.figure.add_subplot(1, 1, 1)

        self.indices = list(range(self.voc.tract_size))
        self.nose_indices = self.indices[-self.voc.nose_size :]

        self.axes.set_ylim(0.0, 5.0)
        self.nose_diameters_line = self.axes.plot(self.nose_indices, nose_diameters, label="nose diameters")[0]
        self.tract_diameters_line = self.axes.plot(self.indices, currect_tract_diameters, label="tract diameters")[0]
        self.axes.legend()

        self.axes.set_title("Tract diameters")
        self.axes.set_xlabel("diameter index")
        self.axes.set_ylabel("diameter [cm]")

        info = self.make_infomation_text()

        self.infomation_text = self.axes.text(1, 4.0, info)

    def update_values(self) -> None:
        """Update values ploted on the figure."""
        nose_diameters = self.voc.nose_diameters
        currect_tract_diameters = self.voc.current_tract_diameters

        info_text = self.make_infomation_text()
        self.tract_diameters_line.set_data(self.indices, currect_tract_diameters)
        self.nose_diameters_line.set_data(self.nose_indices, nose_diameters)

        self.infomation_text.set_text(info_text)

    @staticmethod
    def fig2rgba_array(figure: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy array.

        Args:
            figure (plt.Figure): A matplotlib figure.

        Returns:
            image array (np.ndarray): Numpy array of rendered figure.
                Shape: (Height, Width, RGBA)
        """

        figure.canvas.draw()
        w, h = figure.canvas.get_width_height(physical=True)
        buf = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4).copy()
        buf = np.roll(buf, 3, axis=-1)
        return buf

    @staticmethod
    def fig2rgb_array(figure: plt.Figure) -> np.ndarray:
        """Wrapper of fig2rgba_array. Removing alpha channel."""
        return Renderer.fig2rgba_array(figure)[:, :, :3]

    def render_rgb_array(self) -> np.ndarray:
        """Render figure and return image as numpy array.

        Returns:
            image array (np.ndarray): Rendered image array.
        """

        return self.fig2rgb_array(self.figure)

    def close(self) -> None:
        """Clear figure and close it."""
        self.figure.clear()
        plt.close(self.figure)

    def __del__(self):
        """Destructor"""
        self.close()
