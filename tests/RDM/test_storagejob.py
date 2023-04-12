import unittest
import os

from shutil import rmtree
from moto import mock_s3
import boto3
from pyiron_contrib import Project
from pyiron_contrib.RDM.storagejob import StorageJob
from pyiron_contrib.generic.s3io import FileS3IO
from pyiron_base._tests import TestWithCleanProject
from pyiron_base import FileDataTemplate

full_bucket = "full_bucket"
io_bucket = "io_bucket"
aws_credentials = {"aws_access_key_id": 'fake_access_key',
                   "aws_secret_access_key": 'fake_secret_key'}


class TestStorageJob(TestWithCleanProject):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.project = Project(cls.project_path)
        cls.file1 = os.path.join(cls.project_path, "test_file.txt")
        cls.file2 = os.path.join(cls.project_path, "test_file2.txt")
        with open(cls.file1, 'w') as f:
            f.write('Some text')
        with open(cls.file2, 'w') as f:
            f.write('Some other text')

        cls.moto = mock_s3()
        cls.moto.start()
        cls.res = boto3.resource('s3', **aws_credentials)
        cls.bucket = cls.res.create_bucket(Bucket=full_bucket)
        cls.bucket.put_object(Key='any', Body=b'any text', Metadata={'File_loc': 'root'})
        cls.io_bucket = cls.res.create_bucket(Bucket=io_bucket)

    @classmethod
    def tearDownClass(cls):
        cls.bucket.objects.all().delete()
        cls.bucket.delete()
        cls.io_bucket.objects.all().delete()
        cls.io_bucket.delete()
        cls.moto.stop()
        super().tearDownClass()
        if os.path.exists("tmp_dir"):
            rmtree('tmp_dir')

    def tearDown(self):
        super().tearDown()
        self.io_bucket.objects.all().delete()

    def test_create_storage_job(self):
        job = self.project.create.job.StorageJob('test')
        self.assertIsInstance(job, StorageJob)

    def test___init__(self):
        job = self.project.create.job.StorageJob('test')
        self.assertEqual(job.storage_type, 'local')
        self.assertTrue(job.server.run_mode.interactive)

    def test_use_s3_storage(self):
        # occupy storage of job and remove job without removing the files on s3:
        job = self.project.create.job.StorageJob('test')
        job.use_s3_storage(config=aws_credentials, bucket_name=full_bucket, _only_warn=True)
        job.add_files(self.file1)
        job._hdf5['REQUIRE_FULL_OBJ_FOR_RM'] = False
        # Call remove from the project level
        self.project.remove_job('test')

        job = self.project.create.job.StorageJob('test')
        job2 = self.project.create.job.StorageJob('test2')

        with self.subTest("ValueError: Occupied storage"):
            self.assertRaises(ValueError, job.use_s3_storage, config=aws_credentials, bucket_name=full_bucket)

        with self.subTest("Warning only: Occupied storage"):
            with self.assertLogs(job._logger) as log_watcher:
                job.use_s3_storage(config=aws_credentials, bucket_name=full_bucket, _only_warn=True)
                self.assertIsInstance(job._external_storage, FileS3IO)
                self.assertEqual(len(log_watcher.records), 1)
                self.assertEqual(log_watcher.records[-1].message, "Storage NOT empty - Danger of data loss!")

        with self.subTest("Empty storage, normal case"):
            job.use_s3_storage(config=aws_credentials, bucket_name=io_bucket)
            self.assertIsInstance(job._external_storage, FileS3IO)

        with self.subTest("RuntimeError: Change running StorageJob"):
            job2.run()
            self.assertRaises(RuntimeError, job2.use_s3_storage, config=aws_credentials, bucket_name=io_bucket)

    def test_use_local_storage(self):
        job = self.project.create.job.StorageJob('test')
        job.use_local_storage()

        with self.subTest("switch from s3 to local storage"):
            job.use_s3_storage(config=aws_credentials, bucket_name=io_bucket)
            self.assertIsInstance(job._external_storage, FileS3IO)
            job.use_local_storage()
            self.assertIs(job._external_storage, None)

        with self.subTest("RuntimeError: switch running StorageJob"):
            job.run()
            self.assertRaises(RuntimeError, job.use_local_storage)

    def test_add_files_local(self):
        os.makedirs('tmp_dir', exist_ok=True)
        with open('tmp_dir/test_file.txt', 'w') as f:
            f.write("Changed")

        job = self.project.create.job.StorageJob('test')
        job.add_files(self.file1)
        self.assertEqual(job.files_stored, ["test_file.txt"])
        self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file.txt')))

        with self.subTest(msg="Test copy present file"):
            with self.assertLogs(logger=job._logger, level="WARN") as w:
                job.add_files(self.file1)
                print(w)
                self.assertEqual(len(w.records), 1, msg="Adding the same file again should raise a warning that "
                                                        "the file was not copied")
                self.assertTrue("not copied, since already present" in str(w.records[-1].message))
        with self.subTest(msg="Test copy present file with overwrite=True"):
            job.add_files("tmp_dir/test_file.txt", overwrite=True)
            self.assertEqual(job['test_file.txt'].data, ["Changed"])

        job = self.project.create.job.StorageJob('test2')
        with self.subTest(msg="Test copy list of files"):
            job.add_files([self.file1, self.file2])
            self.assertIn("test_file.txt", job.files_stored)
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file.txt')))
            self.assertIn("test_file2.txt", job.files_stored)
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file2.txt')))
        with self.subTest(msg="Test copy list of identical files"):
            self.assertRaises(ValueError, job.add_files, [self.file1, self.file1])

        rmtree('tmp_dir')

    def test_add_files_s3(self):
        os.makedirs('tmp_dir', exist_ok=True)
        with open('tmp_dir/test_file.txt', 'w') as f:
            f.write("Changed")

        job = self.project.create.job.StorageJob('test')
        job.use_s3_storage(config=aws_credentials, bucket_name=io_bucket)
        job.add_files(self.file1)
        self.assertEqual(job.files_stored, ["test_file.txt"])

        self.assertEqual(['test_file.txt'], job._external_storage.list_nodes())

        with self.subTest(msg="Test copy present file"):
            with self.assertLogs(logger=job._logger, level="WARN") as w:
                job.add_files(self.file1)
                self.assertEqual(len(w.records), 1, msg="Adding the same file again should raise a warning that "
                                                        "the file was not copied")
                self.assertTrue("not copied, since already present" in str(w.records[-1].message))
        with self.subTest(msg="Test copy present file with overwrite=True"):
            job.add_files("tmp_dir/test_file.txt", overwrite=True)
            self.assertEqual(job['test_file.txt'].data, [b"Changed"])  # load_file does not decode bytes, yet.

        job = self.project.create.job.StorageJob('test2')
        job.use_s3_storage(config=aws_credentials, bucket_name=io_bucket)
        with self.subTest(msg="Test copy list of files"):
            job.add_files([self.file1, self.file2])
            self.assertIn("test_file.txt", job.files_stored)
            self.assertIn("test_file.txt", job._external_storage.list_nodes())
            self.assertIn("test_file2.txt", job.files_stored)
            self.assertIn("test_file2.txt", job._external_storage.list_nodes())

        rmtree('tmp_dir')

    def test_reload(self):
        job_local = self.project.create.job.StorageJob('local')
        job_local.add_files(self.file1)

        job_s3 = self.project.create.job.StorageJob('s3')
        job_s3.use_s3_storage(config=aws_credentials, bucket_name=io_bucket)
        job_s3.add_files(self.file1)

        with self.subTest("local"):
            reload_local = self.project['local']
            self.assertEqual(reload_local.files_stored, ["test_file.txt"])
            self.assertTrue(os.path.isfile(os.path.join(reload_local.working_directory, 'test_file.txt')))
        with self.subTest('s3'):
            reload_s3 = self.project['s3']
            self.assertEqual(reload_s3.files_stored, ["test_file.txt"])
            self.assertEqual(reload_s3._external_storage.list_nodes(), ["test_file.txt"])

    def test_remove_files_local(self):
        job = self.project.create.job.StorageJob('test')
        job.add_files([self.file1, self.file2])
        with self.subTest("dryrun"):
            job.remove_files("test_file.txt")
            self.assertTrue("test_file.txt" in job.files_stored)
            self.assertTrue("test_file2.txt" in job.files_stored)
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file.txt')))
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file2.txt')))
        with self.subTest("FileNotFoundError"):
            self.assertRaises(FileNotFoundError, job.remove_files, ["test_file.txt", "test_file2.txt" "no_file.txt"])
            self.assertTrue("test_file.txt" in job.files_stored)
            self.assertTrue("test_file2.txt" in job.files_stored)
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file.txt')))
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file2.txt')))
        with self.subTest("No FileNotFoundError"):
            job.remove_files("no_file.txt", raise_error=False)
            self.assertTrue("test_file.txt" in job.files_stored)
            self.assertTrue("test_file2.txt" in job.files_stored)
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file.txt')))
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file2.txt')))
        with self.subTest("Remove nothing"):
            job.remove_files("no_file.txt", dryrun=False, raise_error=False)
            self.assertTrue("test_file.txt" in job.files_stored)
            self.assertTrue("test_file2.txt" in job.files_stored)
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file.txt')))
            self.assertTrue(os.path.isfile(os.path.join(job.working_directory, 'test_file2.txt')))
        with self.subTest("Remove files"):
            job.remove_files(["test_file.txt", "test_file2.txt", "no_file.txt"], dryrun=False, raise_error=False)
            self.assertFalse("test_file.txt" in job.files_stored)
            self.assertFalse("test_file2.txt" in job.files_stored)
            self.assertFalse(os.path.isfile(os.path.join(job.working_directory, 'test_file.txt')))
            self.assertFalse(os.path.isfile(os.path.join(job.working_directory, 'test_file2.txt')))

    def test_remove_files_s3(self):
        job = self.project.create.job.StorageJob('test')
        job.use_s3_storage(config=aws_credentials, bucket_name=io_bucket)
        job.add_files([self.file1, self.file2])
        with self.subTest("dryrun"):
            job.remove_files("test_file.txt")
            self.assertTrue("test_file.txt" in job.files_stored)
            self.assertTrue("test_file2.txt" in job.files_stored)
            self.assertIn('test_file.txt', job._external_storage.list_nodes())
            self.assertIn('test_file2.txt', job._external_storage.list_nodes())
        with self.subTest("FileNotFoundError"):
            self.assertRaises(FileNotFoundError, job.remove_files, ["test_file.txt", "test_file2.txt" "no_file.txt"])
            self.assertTrue("test_file.txt" in job.files_stored)
            self.assertTrue("test_file2.txt" in job.files_stored)
            self.assertIn('test_file.txt', job._external_storage.list_nodes())
            self.assertIn('test_file2.txt', job._external_storage.list_nodes())
        with self.subTest("No FileNotFoundError"):
            job.remove_files("no_file.txt", raise_error=False)
            self.assertTrue("test_file.txt" in job.files_stored)
            self.assertTrue("test_file2.txt" in job.files_stored)
            self.assertIn('test_file.txt', job._external_storage.list_nodes())
            self.assertIn('test_file2.txt', job._external_storage.list_nodes())
        with self.subTest("Remove nothing"):
            job.remove_files("no_file.txt", dryrun=False, raise_error=False)
            self.assertTrue("test_file.txt" in job.files_stored)
            self.assertTrue("test_file2.txt" in job.files_stored)
            self.assertIn('test_file.txt', job._external_storage.list_nodes())
            self.assertIn('test_file2.txt', job._external_storage.list_nodes())
        with self.subTest("Remove files"):
            job.remove_files(["test_file.txt", "test_file2.txt", "no_file.txt"], dryrun=False, raise_error=False)
            self.assertFalse("test_file.txt" in job.files_stored)
            self.assertFalse("test_file2.txt" in job.files_stored)
            self.assertNotIn('test_file.txt', job._external_storage.list_nodes())
            self.assertNotIn('test_file2.txt', job._external_storage.list_nodes())

    def test___getitem__(self):
        job_local = self.project.create.job.StorageJob('local')
        job_local.add_files(self.file1)

        job_s3 = self.project.create.job.StorageJob('s3')
        job_s3.use_s3_storage(config=aws_credentials, bucket_name=io_bucket)
        job_s3.add_files(self.file1)

        with self.subTest("local"):
            file = job_local["test_file.txt"]
            self.assertIsInstance(file, FileDataTemplate)
            self.assertEqual(file.data, ["Some text"])

        with self.subTest("s3"):
            file = job_s3["test_file.txt"]
            self.assertIsInstance(file, FileDataTemplate)
            self.assertEqual(file.data, [b"Some text"])  # load_file does not decode bytes, yet.


if __name__ == '__main__':
    unittest.main()
