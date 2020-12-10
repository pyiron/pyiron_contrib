import os
from glob import iglob

import ipywidgets as widgets
from IPython import display as IPyDisplay
from IPython.core.display import display
from matplotlib import pylab as plt
from skimage import io
import pandas

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


class _FileBrowser:
    """
        File Browser Widget with S3 support

        Allows to browse files in the local or a remote S3 based file system.
        Selected files may be received from this FileBrowser widget by data attribute.
    """
    def __init__(self,
                 Vbox=None,
                 s3path="",
                 localpath=None,
                 fix_s3_path=False,
                 storage_system="local",
                 fix_storage_sys=False,
                 S3_config_file=None,
                 hdf_as_dirs=False,
                 hide_hdf=False
                 ):
        """
            Filebrowser to browse the local or a remote (S3-based) file system.
            Args:
              s3path (str): Starting path within the remote file system.
              localpath (str/None): Starting path in the local filesystem; if None use current directory.
              fix_s3_path (bool): If True the path in the remote file system cannot be changed.
              storage_system (str): The filesystem to access (fist) either "local" or "S3".
              fix_storage_sys (bool): If True the file system cannot be changed.
              S3_config_file (str): path to a json configuration file with login credentials for the remote file system.
              hdf_as_dirs (bool): If True hdf files in the local file system are shown and treated as directories
              hide_hdf (bool/None): If True hdf files are hidden (also not displayed as directories)
        """
        if Vbox is None:
            self.box = widgets.VBox()
        else:
            self.box = Vbox
        self.fix_s3_path = fix_s3_path
        self.s3path = s3path
        if localpath is None:
            localpath = os.getcwd()
        self.hdf_as_dirs = hdf_as_dirs
        self.hide_hdf = hide_hdf
        self._in_hdf = False
        self._h5_access = None
        self._h5_path = ""
        self.data_sys = storage_system
        if self.data_sys == "local":
            self.path = localpath
        else:
            self.path = self.s3path
        self.output = widgets.Output(layout=widgets.Layout(width='50%', height='100%'))
        self._display_file = DisplayFile(file=None, outwidget=self.output).display_file
        self._display_metadata = DisplayMetadata(metadata=None, outwidget=self.output).display
        self._clickedFiles = []
        self._data = []
        self.fix_storage_sys = fix_storage_sys
        self._path_storage = [localpath, self.s3path]
        self.pathbox = widgets.HBox(layout=widgets.Layout(width='100%', justify_content='flex-start'))
        self.optionbox = widgets.HBox()
        self.filebox = widgets.VBox(layout=widgets.Layout(width='50%', height='100%', justify_content='flex-start'))
        self.path_string_box = widgets.Text(description="(rel) Path", width='min-content')
        try:
            self._s3_access = FileS3IO(config_file=S3_config_file,
                                       group=self.s3path)
            self._update_files()
            self.update()
        except TypeError:
            self._s3_access = None
            self.configure(storage_system="local", fix_storage_sys=True)

    def configure(self,
                 s3path=None,
                 fix_s3_path=None,
                 storage_system=None,
                 fix_storage_sys=None,
                 hdf_as_dirs=None,
                 hide_hdf=None,
                 ):
        """
            Reconfigure and refresh Filebrowser.
            Args:
              s3path (str/None): Path within the remote file system.
              fix_s3_path (bool/None): If True the path in the remote file system cannot be changed.
              storage_system (str/None): The filesystem to access (first): either "local" or "S3".
              fix_storage_sys (bool/None): If True the file system cannot be changed.
              hdf_as_dirs (bool/None): If True hdf files in the local file system are shown and treated as directories
              hide_hdf (bool/None): If True hdf files are hidden (also not displayed as directories)
        """
        if s3path is not None:
            self.s3path = s3path
        if hdf_as_dirs is not None:
            self.hdf_as_dirs = hdf_as_dirs
        if hide_hdf is not None:
            self.hide_hdf = hide_hdf
        if fix_s3_path is not None:
            self.fix_s3_path = fix_s3_path
        if storage_system is not None:
            if storage_system == "S3" and self.data_sys == "local":
                self._path_storage[0] = self.path
                self.path = self._path_storage[1]
            elif storage_system == "local" and self.data_sys == "S3":
                self._path_storage[1] = self.path
                self.path = self._path_storage[0]
            self.data_sys = storage_system
        if fix_storage_sys is not None:
            self.fix_storage_sys = fix_storage_sys

        if s3path is not None:
            if self.data_sys == "S3":
                self.path = self.s3path
            else:
                self._path_storage[1] = self.s3path
        self._update_files()
        self.update()

    def _update_files(self):
        self.files = list()
        self.dirs = list()
        self.h5dirs = list()
        if self.data_sys == "local" and not self._in_hdf:
            if os.path.isdir(self.path):
                for f in iglob(self.path + '/*'):
                    if os.path.isdir(f):
                        self.dirs.append(os.path.split(f)[1])
                    else:
                        filename = os.path.split(f)[1]
                        if os.path.splitext(filename)[1] in ".h5":
                            self.h5dirs.append(filename)
                        else:
                            self.files.append(filename)
        elif self._in_hdf:
            self._h5_access.h5_path = self._h5_path
            list_all_dir = self._h5_access.list_all()
            self.files = list_all_dir["nodes"]
            self.h5dirs = list_all_dir["groups"]
        else:
            if self._s3_access is None:
                raise AttributeError("S3 Access is not set up!")
            self._s3_access.open(self.path)
            self.files = self._s3_access.list_nodes()
            if not self.fix_s3_path:
                self.dirs = self._s3_access.list_groups()

    def gui(self):
        self.update()
        return self.box

    def update(self, Vbox=None):
        if Vbox is None:
            Vbox = self.box
        self._update_files()
        #self._update_pathbox(self.pathbox)
        self._update_optionbox(self.optionbox)
        self._update_filebox(self.filebox)
        body = widgets.HBox([self.filebox, self.output],
                            layout=widgets.Layout(
                                min_height='100px',
                                max_height='800px'
                            ))
        Vbox.children = tuple([self.optionbox, self.pathbox, body])

    def _update_optionbox(self, optionbox):
        def on_sys_change(b):
            if b.description == 'RDM':
                if self.data_sys == 'S3':
                    return
                self._clickedFiles = []
                self._path_storage[0] = self.path
                self.path = self._path_storage[1]
                b.style = checkbox_active_style
                file_sys_button_local.style = checkbox_inactive_style
                if self.fix_s3_path:
                    set_path_button.disabled = True
                self.data_sys = 'S3'
                self._update_files()
                self._update_filebox(self.filebox)
                return
            if b.description == 'local':
                if self.data_sys == 'local':
                    return
                self._clickedFiles = []
                self._path_storage[1] = self.path
                self.path = self._path_storage[0]
                b.style = checkbox_active_style
                set_path_button.disabled = False
                file_sys_button_S3.style = checkbox_inactive_style
                self.data_sys = 'local'
                self._update_files()
                self._update_filebox(self.filebox)
                return
        # some color definitions:
        checkbox_active_style = {"button_color": "#FF8888", 'font_weight': 'bold'}
        checkbox_inactive_style = {"button_color": "#CCAAAA"}

        file_sys_button_local = widgets.Button(description='local', tooltip="Change to local filesystem",
                                         icon="fa-database", layout=widgets.Layout(width='80px'))
        file_sys_button_S3 = widgets.Button(description='RDM', tooltip="Change to Research Data Management System",
                                          icon="fa-database", layout=widgets.Layout(width='80px'))
        if self.data_sys == "local":
            file_sys_button_local.style = checkbox_active_style
            file_sys_button_S3.style = checkbox_inactive_style
        else:
            file_sys_button_local.style = checkbox_inactive_style
            file_sys_button_S3.style = checkbox_active_style

        file_sys_button_local.on_click(on_sys_change)
        file_sys_button_S3.on_click(on_sys_change)

        if self.fix_storage_sys:
            if self.data_sys == "local":
                childs = [file_sys_button_local,  self.path_string_box]
            else:
                childs = [file_sys_button_S3,  self.path_string_box]
        else:
            childs = [file_sys_button_local, file_sys_button_S3, self.path_string_box]

        set_path_button = widgets.Button(description='Set Path', tooltip="Sets current path to provided string.")
        set_path_button.on_click(self._click_option_button)
        if self.fix_s3_path and self.data_sys == "S3":
            set_path_button.disabled = True
        if not (self.fix_s3_path and self.fix_storage_sys and self.data_sys == "S3"):
            childs.append(set_path_button)
        button = widgets.Button(description="Select File(s)", width='min-content',
                                 tooltip='Selects all files ' +
                                         'matching the provided string patten; wildcards allowed.')
        button.on_click(self._click_option_button)
        childs.append(button)
        button = widgets.Button(description="Reset selection", width='min-content')
        button.on_click(self._click_option_button)
        childs.append(button)

        optionbox.children = tuple(childs)

    def _click_option_button(self, b):
        self.output.clear_output(True)
        with self.output:
            print('')
        if b.description == 'Set Path':
            if self.data_sys == 'S3':
                if self.fix_s3_path:
                    return
                path = '/' + self.path
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
            # check path consistency:
            if (self.data_sys == 'local' and os.path.exists(path)):
                self.path = os.path.abspath(path)
            elif (self._s3_access.is_dir(path[1:]) and self.data_sys == 'S3'):
                self.path = path[1:]
            else:
                self.path_string_box.__init__(description="(rel) Path", value='')
                with self.output:
                    if not self.hdf_as_dirs:
                        print('No valid path')
                    else:
                        print('No valid path or path within h5 (not supported)')
                return
            self._update_files()
            self._update_filebox(self.filebox)
            self.path_string_box.__init__(description="(rel) Path", value='')
        if b.description == 'Select File(s)':
            self._select_files()
        if b.description == 'Reset selection':
            self._clickedFiles = []
            self._update_filebox(self.filebox)

    @property
    def data(self):
        if self.data_sys == "S3":
            self._download_data_from_s3()
        else:
            for file in self._clickedFiles:
                data = Data(source=file)
                self._data.append(data)
        with self.output:
            if len(self._data) > 0:
                print('Loaded %i File(s):' % (len(self._data)))
                for i in self._data:
                    print(i.filename)
            else:
                print('No files chosen')
        self._clickedFiles = []
        self._update_filebox(self.filebox)
        return self._data

    def put_data(self, data, metadata=None):
        """
        Uploads a single data object to the current directory of the RDM System
        Args:
            data: MeasuredData Object like the ones stored in self.data
            metadata: metadata to be used (has to be a dictionary of type {"string": "string, })
                      provided metadata overwrites the one possibly present in the data object
        """
        if self._s3_access is None:
            raise AttributeError ("S3 Access is not set up!")
        self._s3_access.put(data, metadata)

    def _download_data_from_s3(self):
        for file in self._clickedFiles:
            filename = os.path.split(file)[1]
            filetype = os.path.splitext(filename)[1]
            if len(filetype[1:]) == 0:
                filetype = None
            else:
                filetype = filetype[1:]
            obj = self._s3_access.get(file, abspath=True)
            data = Data(data=obj['Body'].read(), filename=filename, filetype=filetype,
                        metadata=obj["Metadata"])
            self._data.append(data)

    def upload_data_to_s3(self, files, metadata=None):
        """
        Uploads files into the currently opened directory of the Research Data System
        Arguments:
            files `list` : List of filenames to upload
            metadata `dictionary`: metadata of the files (Not nested, only "str" type)
        """
        if self._s3_access is None:
            raise AttributeError ("S3 Access is not set up!")
        self._s3_access.upload(files=files, metadata=metadata)

    def _select_files(self):
        if len(self.path_string_box.value) == 0:
            path = self.path
        elif self.path_string_box.value[0] != '/':
            path = os.path.join(self.path, self.path_string_box.value)
        elif self.data_sys == "S3":
            with self.output:
                print("Only relative paths supported")
            return
        else:
            path = self.path_string_box.value
        appendlist = []
        if self.data_sys == "local" and not self._in_hdf:
            for f in iglob(path):
                if os.path.isfile(f):
                    appendlist.append(os.path.normpath(f))
        elif self._in_hdf:
            pass
        else:
            appendlist = self._s3_access.glob(path)
        self._clickedFiles.extend(appendlist)
        self._update_filebox(self.filebox)
        with self.output:
            if len(appendlist) > 0:
                print('Selected %i File(s):' % (len(appendlist)))
                for i in appendlist:
                    print(i)
            else:
                print('No additional files selected')

    def _update_pathbox(self, box):
        path_color = '#DDDDAA'
        h5_path_color = '#CCCCAA'
        home_color = '#999999'

        def on_click(b):
            if not b.h5:
                self.path = b.path
                self._h5_access = None
                self._in_hdf = False
            else:
                self._h5_path = b.path
            self._update_files()
            self._update_filebox(self.filebox)
            self.path_string_box.__init__(description="(rel) Path", value='')

        buttons = []
        if self._in_hdf:
            tmppath = self._h5_path
            tmppath_old = self._h5_path + '/'
            while tmppath != tmppath_old:
                tmppath_old = tmppath
                [tmppath, dir] = os.path.split(tmppath)
                if dir == "":
                    dir = self._h5_access.base_name
                button = widgets.Button(description=dir + '/', layout=widgets.Layout(width='auto'))
                button.style.button_color = h5_path_color
                button.path = tmppath_old
                button.h5 = True
                button.on_click(on_click)
                buttons.append(button)
        tmppath = self.path
        tmppath_old = self.path + '/'
        while tmppath != tmppath_old:
            tmppath_old = tmppath
            [tmppath, dir] = os.path.split(tmppath)
            button = widgets.Button(description=dir + '/', layout=widgets.Layout(width='auto'))
            button.style.button_color = path_color
            button.path = tmppath_old
            button.h5 = False
            button.on_click(on_click)
            if self.fix_s3_path and self.data_sys == "S3":
                button.disabled = True
            buttons.append(button)
        button = widgets.Button(icon="fa-home", layout=widgets.Layout(width='auto'))
        button.style.button_color = home_color
        if self.data_sys == 'local':
            button.path = os.getcwd()
        else:
            button.path = self.s3path
            if self.fix_s3_path:
                button.disabled = True
        button.h5 = False
        button.on_click(on_click)
        buttons.append(button)
        buttons.reverse()
        box.children = tuple(buttons)

    def _update_filebox(self, filebox):
        # color definitions
        dir_color = '#9999FF'
        h5_dir_color = '#9999EE'
        file_chosen_color = '#FFBBBB'
        file_color = '#DDDDDD'
        self.output.clear_output(True)

        def on_click(b):
            if b.h5:
                self._h5_path = "/"
                self._in_hdf = True
                self._h5_access = FileHDFio(file_name=os.path.join(self.path, b.h5file),
                                            h5_path=self._h5_path,
                                            mode="r")
            else:
                self.path = os.path.join(self.path, b.description)
            self._update_files()
            self._update_filebox(filebox)

        def on_click_h5(b):
            self._h5_path = os.path.join(self._h5_path, b.description)
            self._update_files()
            self._update_filebox(filebox)

        def on_click_file(b):
            f = os.path.join(self.path, b.description)
            self.output.clear_output(True)
            if self.data_sys == 'local' and not self._in_hdf:
                self._display_file(f)
            elif self._in_hdf:
                with self.output:
                    print(self._h5_access[b.description])
            else:
                metadata = self._s3_access.get_metadata(f, abspath=True)
                self._display_metadata(metadata)
            if f in self._clickedFiles:
                b.style.button_color = file_color
                self._clickedFiles.remove(f)
            elif self._in_hdf:
                pass
            else:
                b.style.button_color = file_chosen_color
                self._clickedFiles.append(f)

        buttons = []
        item_layout = widgets.Layout(width='80%',
                                     height='30px',
                                     min_height='24px',
                                     display='flex',
                                     align_items="center",
                                     justify_content='flex-start')
        for f in self.dirs:
            button = widgets.Button(description=f,
                                    icon="fa-folder",
                                    layout=item_layout)
            button.style.button_color = dir_color
            button.h5 = False
            button.on_click(on_click)
            buttons.append(button)

        if not self.hide_hdf and self.hdf_as_dirs:
            for f in self.h5dirs:
                button = widgets.Button(description=os.path.splitext(f)[0],
                                        icon="fa-folder",
                                        layout=item_layout)
                button.style.button_color = h5_dir_color
                if not self._in_hdf:
                    button.h5 = True
                    button.h5file = f
                    button.on_click(on_click)
                else:
                    button.on_click(on_click_h5)
                buttons.append(button)
        elif not self.hide_hdf:
            for f in self.h5dirs:
                button = widgets.Button(description=f,
                                        icon="fa-file-o",
                                        layout=item_layout)
                if os.path.join(self.path, f) in self._clickedFiles:
                    button.style.button_color = file_chosen_color
                else:
                    button.style.button_color = file_color
                button.on_click(on_click_file)
                buttons.append(button)

        for f in self.files:
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
        self._update_pathbox(self.pathbox)


class FileBrowser(_FileBrowser):
    """
        File Browser Widget with S3 support

        Allows to browse files in the local or a remote S3 based file system.
        Selected files may be received from this FileBrowser widget by its data attribute.
    """
    def __init__(self,
                 project=None,
                 Vbox = None,
                 s3path="",
                 fix_s3_path=False,
                 storage_system="local",
                 fix_storage_sys=False,
                 hdf_as_dirs=False,
                 S3_config_file = None
                 ):

        path = project.path[:-1] if project is not None else None
        super().__init__(Vbox=Vbox,
                         s3path=s3path,
                         localpath=path,
                         fix_s3_path=fix_s3_path,
                         hdf_as_dirs=hdf_as_dirs,
                         storage_system=storage_system,
                         fix_storage_sys=fix_storage_sys,
                         S3_config_file=S3_config_file)
    __init__.__doc__ = _FileBrowser.__init__.__doc__

