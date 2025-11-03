import os
import shlex
import tarfile
import tempfile
import uuid
from abc import abstractmethod
from pathlib import Path

from modules.cloud.BaseFileSync import BaseFileSync
from modules.util.config.CloudConfig import CloudConfig, CloudSecretsConfig

import fabric


class BaseSSHFileSync(BaseFileSync):
    def __init__(self, config: CloudConfig, secrets: CloudSecretsConfig):
        super().__init__(config, secrets)
        self.sync_connection=fabric.Connection(host=secrets.host,port=secrets.port,user=secrets.user)

    def close(self):
        if self.sync_connection:
            self.sync_connection.close()

    @abstractmethod
    def upload_files(self,local_files,remote_dir: Path):
        pass

    @abstractmethod
    def download_files(self,local_dir: Path,remote_files):
        pass

    @abstractmethod
    def upload_file(self,local_file: Path,remote_file: Path):
        pass

    @abstractmethod
    def download_file(self,local_file: Path,remote_file: Path):
        pass

    def sync_up_file(self,local : Path,remote : Path):
        sync_info=self.__get_sync_info(remote)
        if not self.__needs_upload(local=local,remote=remote,sync_info=sync_info):
            return

        self.sync_connection.open()
        self.sync_connection.run(f'mkdir -p {shlex.quote(remote.parent.as_posix())}',in_stream=False)
        self.upload_file(local_file=local,remote_file=remote)


    def sync_up_dir(self,local : Path,remote: Path,recursive: bool,sync_info=None):
        # Always pack directories for faster upload
        try:
            # Create tar.gz archive(s) locally (may be sharded if very large)
            archive_paths = self._pack_directory(local, recursive)
            
            # If single archive, wrap in list for consistent handling
            if isinstance(archive_paths, Path):
                archive_paths = [archive_paths]
            
            self.sync_connection.open()
            self.sync_connection.run(f'mkdir -p {shlex.quote(remote.parent.as_posix())}',in_stream=False)
            
            # Upload all archive shards
            remote_archives = []
            for i, archive_path in enumerate(archive_paths):
                if len(archive_paths) == 1:
                    # Single archive: use simple name
                    remote_archive = remote.parent / f"{remote.name}.tar.gz"
                else:
                    # Multiple shards: use numbered pattern
                    remote_archive = remote.parent / f"{remote.name}.tar.gz.part{i+1:03d}"
                remote_archives.append(remote_archive)
                
                print(f"Uploading archive {i+1}/{len(archive_paths)} ({archive_path.stat().st_size / (1024*1024):.1f} MB)...")
                self.upload_file(local_file=archive_path, remote_file=remote_archive)
            
            # Unpack all archives on remote
            self._unpack_archives(remote_archives, remote, recursive)
            
            # Clean up remote archives
            for remote_archive in remote_archives:
                self.sync_connection.run(f'rm -f {shlex.quote(remote_archive.as_posix())}', in_stream=False, warn=True)
            
            # Clean up local archives
            for archive_path in archive_paths:
                archive_path.unlink()
            
        except Exception as e:
            # Fall back to file-by-file upload if packing fails
            print(f"Warning: Packing failed ({e}), falling back to file-by-file upload...")
            if sync_info is None:
                sync_info=self.__get_sync_info(remote)
            self.sync_connection.open()
            self.sync_connection.run(f'mkdir -p {shlex.quote(remote.as_posix())}',in_stream=False)
            files=[]
            for local_entry in local.iterdir():
                if local_entry.is_file():
                    remote_entry=remote/local_entry.name
                    if self.__needs_upload(local=local_entry,remote=remote_entry,sync_info=sync_info):
                        files.append(local_entry)
                elif recursive and local_entry.is_dir():
                    self.sync_up_dir(local=local_entry,remote=remote/local_entry.name,recursive=True,sync_info=sync_info)

            self.upload_files(local_files=files,remote_dir=remote)

    def sync_down_file(self,local : Path,remote : Path):
        sync_info=self.__get_sync_info(remote)
        if not self.__needs_download(local=local,remote=remote,sync_info=sync_info):
            return
        local.parent.mkdir(parents=True,exist_ok=True)
        self.download_file(local_file=local,remote_file=remote)

    def sync_down_dir(self,local : Path,remote : Path,filter=None):
        sync_info=self.__get_sync_info(remote)
        dirs={}
        for remote_entry in sync_info:
            local_entry=local / remote_entry.relative_to(remote)
            if ((filter is not None and not filter(remote_entry))
                or not self.__needs_download(local=local_entry,remote=remote_entry,sync_info=sync_info)):
                continue

            if local_entry.parent not in dirs:
                dirs[local_entry.parent]=[]
            dirs[local_entry.parent].append(remote_entry)

        for dir,files in dirs.items():
            dir.mkdir(parents=True,exist_ok=True)
            self.download_files(local_dir=dir,remote_files=files)


    def __get_sync_info(self,remote : Path):
        cmd = f'find {shlex.quote(remote.as_posix())} -type f -exec stat --printf "%n\\t%s\\t%Y\\n"' + ' {} \\;'
        self.sync_connection.open()
        result=self.sync_connection.run(cmd,warn=True,hide=True,in_stream=False)
        info={}
        for line in result.stdout.splitlines():
            sp=line.split('\t')
            info[Path(sp[0])]={
                    'size': int(sp[1]),
                    'mtime': int(sp[2])
                }
        return info

    @staticmethod
    def __needs_upload(local : Path,remote : Path,sync_info):
        return (
            remote not in sync_info
            or local.stat().st_size != sync_info[remote]['size']
            or local.stat().st_mtime > sync_info[remote]['mtime']
        )

    @staticmethod
    def __needs_download(local : Path,remote : Path,sync_info):
        return (
            not local.exists()
            or remote not in sync_info
            or local.stat().st_size != sync_info[remote]['size']
            or local.stat().st_mtime < sync_info[remote]['mtime']
        )

    def _get_directory_size(self, local_dir: Path) -> int:
        """
        Calculate total size of directory in bytes.
        
        Args:
            local_dir: Directory to calculate size for
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(local_dir):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                try:
                    total_size += filepath.stat().st_size
                except (OSError, FileNotFoundError):
                    # Skip files that can't be accessed
                    pass
        return total_size

    def _pack_directory(self, local_dir: Path, recursive: bool) -> Path | list[Path]:
        """
        Pack a directory into one or more tar.gz archives (sharded if large).
        
        Args:
            local_dir: Local directory to pack
            recursive: Whether to include subdirectories (ignored, always recursive)
            
        Returns:
            Path to single archive file, or list of Paths if sharded
        """
        # Shard size threshold: 2GB per archive
        SHARD_SIZE_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2GB
        
        # Get all files to pack
        all_files = []
        for item in local_dir.iterdir():
            all_files.append(item)
        
        if not all_files:
            # Empty directory - create empty archive
            temp_dir = Path(tempfile.gettempdir())
            unique_id = uuid.uuid4().hex[:8]
            archive_name = f"{local_dir.name}_{unique_id}.tar.gz"
            archive_path = temp_dir / archive_name
            with tarfile.open(archive_path, 'w:gz') as tar:
                pass  # Create empty archive
            return archive_path
        
        # Calculate total size to decide if sharding is needed
        total_size = self._get_directory_size(local_dir)
        
        # If total size is below threshold, create single archive
        if total_size < SHARD_SIZE_THRESHOLD:
            temp_dir = Path(tempfile.gettempdir())
            unique_id = uuid.uuid4().hex[:8]
            archive_name = f"{local_dir.name}_{unique_id}.tar.gz"
            archive_path = temp_dir / archive_name
            
            with tarfile.open(archive_path, 'w:gz') as tar:
                for item in all_files:
                    tar.add(item, arcname=item.name, recursive=True)
            
            return archive_path
        
        # Sharding needed: split files across multiple archives
        print(f"Directory size ({total_size / (1024*1024*1024):.2f} GB) exceeds threshold, creating sharded archives...")
        temp_dir = Path(tempfile.gettempdir())
        unique_id = uuid.uuid4().hex[:8]
        archive_paths = []
        
        current_shard = []
        current_shard_size = 0
        shard_num = 1
        
        def create_shard(shard_files, num):
            """Create a single archive shard from a list of files."""
            archive_name = f"{local_dir.name}_{unique_id}.tar.gz.part{num:03d}"
            archive_path = temp_dir / archive_name
            with tarfile.open(archive_path, 'w:gz') as tar:
                for item in shard_files:
                    tar.add(item, arcname=item.name, recursive=True)
            return archive_path
        
        for item in all_files:
            # Get item size (for directories, use directory size)
            if item.is_file():
                item_size = item.stat().st_size
            else:
                item_size = self._get_directory_size(item)
            
            # If single item exceeds threshold, put it in its own shard
            if item_size >= SHARD_SIZE_THRESHOLD:
                # Save current shard if it has files
                if current_shard:
                    archive_paths.append(create_shard(current_shard, shard_num))
                    shard_num += 1
                    current_shard = []
                    current_shard_size = 0
                
                # Create shard for this large item
                archive_paths.append(create_shard([item], shard_num))
                shard_num += 1
            elif current_shard_size + item_size > SHARD_SIZE_THRESHOLD:
                # Current shard would exceed threshold, create it and start new one
                archive_paths.append(create_shard(current_shard, shard_num))
                shard_num += 1
                current_shard = [item]
                current_shard_size = item_size
            else:
                # Add to current shard
                current_shard.append(item)
                current_shard_size += item_size
        
        # Create final shard if it has files
        if current_shard:
            archive_paths.append(create_shard(current_shard, shard_num))
        
        print(f"Created {len(archive_paths)} archive shard(s)")
        return archive_paths

    def _unpack_archives(self, remote_archives: list[Path], remote_target: Path, recursive: bool):
        """
        Unpack one or more tar.gz archives on the remote side.
        
        Args:
            remote_archives: List of archive paths on remote (or single archive will be wrapped)
            remote_target: Target directory where contents should be extracted
            recursive: Whether recursive unpacking (ignored, always recursive)
        """
        # Ensure target directory exists
        self.sync_connection.run(f'mkdir -p {shlex.quote(remote_target.as_posix())}', in_stream=False)
        
        # Extract all archives in order
        for i, remote_archive in enumerate(remote_archives):
            print(f"Extracting archive {i+1}/{len(remote_archives)}...")
            # Extract archive contents directly to target directory
            # Using -C to extract to target directory
            cmd = f'tar -xzf {shlex.quote(remote_archive.as_posix())} -C {shlex.quote(remote_target.as_posix())}'
            self.sync_connection.run(cmd, in_stream=False)
