import functools
import os
from glob import iglob

import ipywidgets as widgets
from IPython import display as IPyDisplay
from IPython.core.display import display
from matplotlib import pylab as plt
from skimage import io
import pandas

from pyiron_base import Project as BaseProject
from pyiron_base.generic.hdfio import FileHDFio
from pyiron_contrib.generic.s3io import FileS3IO
from pyiron_contrib.generic.data import Data


class DisplayFile:

    """ Class to display a file in the given outwidget. """
    def __init__(self, file, outwidget):
        """
            Class to display different files in a notebook.

            Args:
                file (str/None): path to the file to be displayed.
                outwidget (:class:`ipywidgets.Output` widget): Will be used to display the file.
        """
        self.output = outwidget
        self.fig = None
        self.ax = None
        self.file = file
        if file is not None:
            self._display_file()

    def display_file(self, file, outwidget=None):
        """
            Display the file in the outwidget,

            Args:
                file (str): path to the file to be displayed.
                outwidget (:class:`ipywidgets.Output` widget / None): New output widget to be used to display the file.
        """
        if outwidget is not None:
            self.output = outwidget
        self.file = file
        self._display_file()

    def _display_file(self):
        _, filetype = os.path.splitext(self.file)
        if filetype.lower() in ['.tif', '.tiff']:
            self._display_tiff()
        elif filetype.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            self._display_img()
        elif filetype.lower() in ['.txt']:
            self._display_txt()
        elif filetype.lower() in ['.csv']:
            self._display_csv()
        else:
            self._display_default()

    def _display_tiff(self):
        plt.ioff()
        data = io.imread(self.file)
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()
        self.ax.imshow(data)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        with self.output:
            display(self.fig)

    def _display_txt(self):
        with self.output:
            with open(self.file) as f:
                print(f.read(), end='')

    def _display_csv(self):
        with self.output:
            display(pandas.read_csv(self.file))

    def _display_img(self):
        with self.output:
            display(IPyDisplay.Image(self.file))

    def _display_default(self):
        try:
            with self.output:
                display(self.file)
        except:
            with self.output:
                print(self.file)


class DisplayMetadata:

    """ Class to display metadata of a file in the given outwidget. """
    def __init__(self, metadata, outwidget):
        """
            Display the metadata in the outwidget.

            Args:
                metadata (dict/None): Metadata to be displayed.
                outwidget (:class:`ipywidgets.Output` widget): New output widget to be used to display the metadata.
        """
        self.output = outwidget
        self.metadata = metadata
        if metadata is not None:
            self._display_metadata()

    def display(self, metadata, outwidget=None):
        """
            Display the metadata in the outwidget

            Args:
                metadata (dict): Metadata to be displayed.
                outwidget (:class:`ipywidgets.Output` widget / None): New output widget to be used to display the metadata.
        """
        self.metadata = metadata
        if outwidget is not None:
            self.output = outwidget
        self._display_metadata()

    def _display_metadata(self):
        with self.output:
            print("Metadata:")
            print("------------------------")
            for key, value in self.metadata.items():
                print(key + ': ' + value)


class ProjectBrowser:
    """
        File Browser Widget with S3 support

        Allows to browse files in the local or a remote S3 based file system.
        Selected files may be received from this FileBrowser widget by data attribute.
    """
    def __init__(self,
                 project,
                 Vbox=None,
                 fix_path=False,
                 show_files=True
                 ):
        """
            Filebrowser to browse the project file system.
            Args:
                project (:class:`pyiron.Project`):
                fix_path (bool): If True the path in the file system cannot be changed.
        """
        self.project = project.copy()
        self._node_as_dirs = isinstance(self.project, BaseProject)
        self._initial_path = self.path
        self._project_root_path = self.project.root_path

        if Vbox is None:
            self.box = widgets.VBox()
        else:
            self.box = Vbox
        self.fix_path = fix_path
        self._busy = False
        self._show_files = show_files
        self.output = widgets.Output(layout=widgets.Layout(width='50%', height='100%'))
        #TODO: move to project self._display_file = DisplayFile(file=None, outwidget=self.output).display_file
        #TODO: move to project self._display_metadata = DisplayMetadata(metadata=None, outwidget=self.output).display
        self._clickedFiles = []
        self._data = []
        self.pathbox = widgets.HBox(layout=widgets.Layout(width='100%', justify_content='flex-start'))
        self.optionbox = widgets.HBox()
        self.filebox = widgets.VBox(layout=widgets.Layout(width='50%', height='100%', justify_content='flex-start'))
        self.path_string_box = widgets.Text(description="(rel) Path", width='min-content')
        self.update()

    @property
    def path(self):
        return self.project.path

    def _busy_check(self, busy=True):
        if self._busy and busy:
            return
        else:
            self._busy = busy

    def _update_files(self):
        # HDF and S3 project do not have list_files
        self.files = list()
        if self._show_files:
            try:
                self.files = self.project.list_files()
            except AttributeError:
                pass
        self.nodes = self.project.list_nodes()
        self.dirs = self.project.list_groups()

    def gui(self):
        self.update()
        return self.box

    def update(self, Vbox=None):
        if Vbox is None:
            Vbox = self.box
        self.output.clear_output(True)
        self._update_files()
        self._update_optionbox(self.optionbox)
        self._update_filebox(self.filebox)
        self._update_pathbox(self.pathbox)
        body = widgets.HBox([self.filebox, self.output],
                            layout=widgets.Layout(
                                min_height='100px',
                                max_height='800px'
                            ))
        Vbox.children = tuple([self.optionbox, self.pathbox, body])

    def _update_optionbox(self, optionbox):

        set_path_button = widgets.Button(description='Set Path', tooltip="Sets current path to provided string.")
        set_path_button.on_click(self._click_option_button)
        if self.fix_path:
            set_path_button.disabled = True
        childs = [set_path_button, self.path_string_box]

        #button = widgets.Button(description="Select File(s)", width='min-content',
        #                         tooltip='Selects all files ' +
        #                                 'matching the provided string patten; wildcards allowed.')
        #button.on_click(self._click_option_button)
        #childs.append(button)
        button = widgets.Button(description="Reset selection", width='min-content')
        button.on_click(self._click_option_button)
        childs.append(button)

        optionbox.children = tuple(childs)

    def _click_option_button(self, b):
        self._busy_check()
        self.output.clear_output(True)
        with self.output:
            print('')
        if b.description == 'Set Path':
            if self.fix_path:
                return
            else:
                path = self.path
            if len(self.path_string_box.value) == 0:
                with self.output:
                    print('No path given')
                return
            elif self.path_string_box.value[0] != '/':
                path = path + '/' + self.path_string_box.value
            else:
                path = self.path_string_box.value
            self._update_project(path)
        if b.description == 'Reset selection':
            self._clickedFiles = []
            self._update_filebox(self.filebox)
        self._busy_check(False)

    #@property
    #def data(self):
    #    for file in self._clickedFiles:
    #        data = Data(source=file)
    #        self._data.append(data)
    #    with self.output:
    #        if len(self._data) > 0:
    #            print('Loaded %i File(s):' % (len(self._data)))
    #            for i in self._data:
    #                print(i.filename)
    #        else:
    #            print('No files chosen')
    #    self._clickedFiles = []
    #    self._update_filebox(self.filebox)
    #    return self._data

    def _update_project_worker(self, rel_path):
        try:
            new_project = self.project[rel_path]
        except ValueError:
            self.path_string_box.__init__(description="(rel) Path", value='')
            return "No valid path"
        else:
            if new_project is not None:
                self.project = new_project

    def _update_project(self, path):
        self.output.clear_output(True)
        # check path consistency:
        rel_path = os.path.relpath(path, self.path)
        if rel_path == '.':
            return
        self._update_project_worker(rel_path)
        if self.path != path:
            rel_path = os.path.relpath(path, self.path)
            self._update_project_worker(rel_path)
        self._node_as_dirs = isinstance(self.project, BaseProject)
        self._update_files()
        self._update_filebox(self.filebox)
        self._update_pathbox(self.pathbox)
        self.path_string_box.__init__(description="(rel) Path", value='')

    def _update_pathbox(self, box):
        path_color = '#DDDDAA'
        h5_path_color = '#CCCCAA'
        home_color = '#999999'

        def on_click(b):
            self._busy_check()
            self._update_project(b.path)
            self._busy_check(False)

        #with self.output:
        #    display(self.project)
        buttons = []
        tmppath = os.path.abspath(self.path)
        if tmppath[-1] == '/':
            tmppath = tmppath[:-1]
        len_root_path = len(os.path.split(self._project_root_path[:-1])[0])
        tmppath_old = tmppath + '/'
        while tmppath != tmppath_old:
            tmppath_old = tmppath
            [tmppath, curentdir] = os.path.split(tmppath)
            button = widgets.Button(description=curentdir + '/', layout=widgets.Layout(width='auto'))
            button.style.button_color = path_color
            button.path = tmppath_old
            button.on_click(on_click)
            if self.fix_path or len(tmppath) < len_root_path - 1:
                button.disabled = True
            buttons.append(button)
        button = widgets.Button(icon="fa-home", layout=widgets.Layout(width='auto'))
        button.style.button_color = home_color
        button.path = self._initial_path
        if self.fix_path:
            button.disabled = True
        button.on_click(on_click)
        buttons.append(button)
        buttons.reverse()
        box.children = tuple(buttons)

    def _update_filebox(self, filebox):
        # color definitions
        dir_color = '#9999FF'
        node_color = '#9999EE'
        file_chosen_color = '#FFBBBB'
        file_color = '#DDDDDD'

        if self._node_as_dirs:
            dirs = self.dirs + self.nodes
            files = self.files
        else:
            files = self.nodes + self.files
            dirs = self.dirs

        def on_click_group(b):
            self._busy_check()
            path = os.path.join(self.path, b.description)
            self._update_project(path)
            self._busy_check(False)

        def on_click_file(b):
            self._busy_check()
            f = os.path.join(self.path, b.description)
            self.output.clear_output(True)
            with self.output:
                try:
                    display(self.project[b.description])
                except:
                    print([b.description])
            if f in self._clickedFiles:
                b.style.button_color = file_color
                self._clickedFiles.remove(f)
            else:
                b.style.button_color = file_chosen_color
                #self._clickedFiles.append(f)
                self._clickedFiles = [f]
                self._update_filebox(filebox)
            self._busy_check(False)

        buttons = []
        item_layout = widgets.Layout(width='80%',
                                     height='30px',
                                     min_height='24px',
                                     display='flex',
                                     align_items="center",
                                     justify_content='flex-start')

        for f in dirs:
            button = widgets.Button(description=f,
                                    icon="fa-folder",
                                    layout=item_layout)
            button.style.button_color = dir_color
            button.on_click(on_click_group)
            buttons.append(button)

        for f in files:
            button = widgets.Button(description=f,
                                    icon="fa-file-o",
                                    layout=item_layout)
            if os.path.join(self.path, f) in self._clickedFiles:
                button.style.button_color = file_chosen_color
            else:
                button.style.button_color = file_color
            button.on_click(on_click_file)
            buttons.append(button)

        filebox.children = tuple(buttons)

