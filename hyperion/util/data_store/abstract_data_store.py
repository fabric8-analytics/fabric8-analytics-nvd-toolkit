import abc


class AbstractDataStore(object):

    @abc.abstractmethod
    def get_name(self):
        return

    @abc.abstractmethod
    def list_files(self, prefix=None, max_count=None):
        """List all the files in the source directory"""
        return

    @abc.abstractmethod
    def read_json_file(self, filename):
        """Read JSON file from the data source"""
        return

    @abc.abstractmethod
    def read_all_json_files(self):
        """Read all the files from the data source"""
        return

    @abc.abstractmethod
    def write_json_file(self, filename, contents):
        """Write JSON file into data source"""
        return

    @abc.abstractmethod
    def upload_file(self, src, target):
        """Upload file into data store"""
        return

    @abc.abstractmethod
    def download_file(self, src, target):
        """Download file from data store"""
        return
