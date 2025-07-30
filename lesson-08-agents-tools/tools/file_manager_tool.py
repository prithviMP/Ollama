#!/usr/bin/env python3
"""
File Manager Tool for Agent File Operations

Provides safe file system operations for agents including reading, writing,
listing directories, and managing files within a sandboxed environment.
"""

import os
import json
import csv
import shutil
from typing import Any, Optional, Dict, List, Union
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class FileManagerTool(BaseTool):
    """Comprehensive file management tool with safety controls."""
    
    name = "file_manager"
    description = """
    Perform file system operations including:
    - read_file(path): Read text file contents
    - write_file(path, content): Write content to file
    - list_dir(path): List directory contents
    - create_dir(path): Create directory
    - delete_file(path): Delete file (with confirmation)
    - copy_file(source, destination): Copy file
    - move_file(source, destination): Move file
    - file_info(path): Get file information
    - search_files(directory, pattern): Search for files
    
    Input format: "operation:path:content" (content optional)
    Examples: "read_file:/tmp/data.txt", "write_file:/tmp/output.txt:Hello World"
    Returns: Operation result with file information
    """
    
    def __init__(self, workspace_dir: str = "./agent_workspace", max_file_size: int = 10485760):
        super().__init__()
        self.workspace_dir = Path(workspace_dir).resolve()
        self.max_file_size = max_file_size  # 10MB default
        self.allowed_extensions = {
            '.txt', '.json', '.csv', '.md', '.log', '.yml', '.yaml', 
            '.xml', '.html', '.css', '.js', '.py', '.sh', '.sql'
        }
        
        # Create workspace if it doesn't exist
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Security: Ensure we don't operate outside workspace
        self.safe_mode = True
    
    def _run(self, command: str) -> str:
        """Execute file management command."""
        try:
            parts = command.split(':', 2)
            if len(parts) < 2:
                return "Error: Invalid command format. Use 'operation:path' or 'operation:path:content'"
            
            operation = parts[0].strip()
            path = parts[1].strip()
            content = parts[2] if len(parts) > 2 else ""
            
            # Security check
            if not self._is_safe_path(path):
                return f"Error: Access denied to path outside workspace: {path}"
            
            # Route to appropriate method
            if operation == "read_file":
                return self._read_file(path)
            elif operation == "write_file":
                return self._write_file(path, content)
            elif operation == "list_dir":
                return self._list_directory(path)
            elif operation == "create_dir":
                return self._create_directory(path)
            elif operation == "delete_file":
                return self._delete_file(path)
            elif operation == "copy_file":
                if not content:
                    return "Error: copy_file requires destination path in content field"
                return self._copy_file(path, content)
            elif operation == "move_file":
                if not content:
                    return "Error: move_file requires destination path in content field"
                return self._move_file(path, content)
            elif operation == "file_info":
                return self._get_file_info(path)
            elif operation == "search_files":
                pattern = content if content else "*"
                return self._search_files(path, pattern)
            else:
                return f"Error: Unknown operation '{operation}'. Available: read_file, write_file, list_dir, create_dir, delete_file, copy_file, move_file, file_info, search_files"
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if path is within workspace directory."""
        if not self.safe_mode:
            return True
        
        try:
            resolved_path = Path(path).resolve()
            # Check if path is within workspace
            return str(resolved_path).startswith(str(self.workspace_dir))
        except:
            return False
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to workspace."""
        if Path(path).is_absolute():
            return Path(path)
        else:
            return self.workspace_dir / path
    
    def _read_file(self, path: str) -> str:
        """Read file contents."""
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return f"Error: File not found: {path}"
            
            if not file_path.is_file():
                return f"Error: Path is not a file: {path}"
            
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                return f"Error: File too large (max {self.max_file_size} bytes): {path}"
            
            # Check file extension
            if file_path.suffix.lower() not in self.allowed_extensions:
                return f"Error: File type not allowed: {file_path.suffix}"
            
            # Read file based on extension
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return f"JSON file contents:\\n{json.dumps(data, indent=2)}"
            
            elif file_path.suffix.lower() == '.csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                return f"CSV file contents ({len(rows)} rows):\\n" + "\\n".join([",".join(row) for row in rows[:10]])
            
            else:
                # Read as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return f"File contents ({len(content)} characters):\\n{content}"
        
        except UnicodeDecodeError:
            return f"Error: Cannot read file as text (binary file?): {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _write_file(self, path: str, content: str) -> str:
        """Write content to file."""
        try:
            file_path = self._resolve_path(path)
            
            # Check file extension
            if file_path.suffix.lower() not in self.allowed_extensions:
                return f"Error: File type not allowed: {file_path.suffix}"
            
            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check content size
            if len(content.encode('utf-8')) > self.max_file_size:
                return f"Error: Content too large (max {self.max_file_size} bytes)"
            
            # Write file based on extension
            if file_path.suffix.lower() == '.json':
                try:
                    # Validate JSON
                    data = json.loads(content)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON content: {str(e)}"
            else:
                # Write as text
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Get file info
            file_size = file_path.stat().st_size
            return f"✅ File written successfully: {path}\\nSize: {file_size} bytes\\nContent length: {len(content)} characters"
        
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _list_directory(self, path: str) -> str:
        """List directory contents."""
        try:
            dir_path = self._resolve_path(path)
            
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            
            if not dir_path.is_dir():
                return f"Error: Path is not a directory: {path}"
            
            items = []
            total_size = 0
            
            for item in sorted(dir_path.iterdir()):
                try:
                    stat = item.stat()
                    size = stat.st_size
                    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    
                    if item.is_dir():
                        items.append(f"📁 {item.name}/ - {modified}")
                    else:
                        items.append(f"📄 {item.name} ({size} bytes) - {modified}")
                        total_size += size
                
                except (OSError, PermissionError):
                    items.append(f"❌ {item.name} (access denied)")
            
            if not items:
                return f"Directory is empty: {path}"
            
            result = f"Directory listing for: {path}\\n"
            result += f"Total items: {len(items)} | Total size: {total_size} bytes\\n\\n"
            result += "\\n".join(items)
            
            return result
        
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def _create_directory(self, path: str) -> str:
        """Create directory."""
        try:
            dir_path = self._resolve_path(path)
            
            if dir_path.exists():
                return f"Directory already exists: {path}"
            
            dir_path.mkdir(parents=True, exist_ok=True)
            return f"✅ Directory created: {path}"
        
        except Exception as e:
            return f"Error creating directory: {str(e)}"
    
    def _delete_file(self, path: str) -> str:
        """Delete file or directory."""
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return f"Error: Path not found: {path}"
            
            if file_path.is_dir():
                # Count items in directory
                items = list(file_path.iterdir())
                if items:
                    return f"Error: Directory not empty ({len(items)} items). Use force deletion if needed: {path}"
                file_path.rmdir()
                return f"✅ Empty directory deleted: {path}"
            else:
                file_size = file_path.stat().st_size
                file_path.unlink()
                return f"✅ File deleted: {path} ({file_size} bytes)"
        
        except Exception as e:
            return f"Error deleting: {str(e)}"
    
    def _copy_file(self, source: str, destination: str) -> str:
        """Copy file."""
        try:
            src_path = self._resolve_path(source)
            dst_path = self._resolve_path(destination)
            
            if not src_path.exists():
                return f"Error: Source file not found: {source}"
            
            if not src_path.is_file():
                return f"Error: Source is not a file: {source}"
            
            # Create destination directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            
            src_size = src_path.stat().st_size
            dst_size = dst_path.stat().st_size
            
            return f"✅ File copied successfully\\nSource: {source} ({src_size} bytes)\\nDestination: {destination} ({dst_size} bytes)"
        
        except Exception as e:
            return f"Error copying file: {str(e)}"
    
    def _move_file(self, source: str, destination: str) -> str:
        """Move file."""
        try:
            src_path = self._resolve_path(source)
            dst_path = self._resolve_path(destination)
            
            if not src_path.exists():
                return f"Error: Source file not found: {source}"
            
            # Create destination directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            file_size = src_path.stat().st_size
            shutil.move(str(src_path), str(dst_path))
            
            return f"✅ File moved successfully\\nFrom: {source}\\nTo: {destination} ({file_size} bytes)"
        
        except Exception as e:
            return f"Error moving file: {str(e)}"
    
    def _get_file_info(self, path: str) -> str:
        """Get detailed file information."""
        try:
            file_path = self._resolve_path(path)
            
            if not file_path.exists():
                return f"Error: Path not found: {path}"
            
            stat = file_path.stat()
            
            info = f"File Information: {path}\\n"
            info += f"Type: {'Directory' if file_path.is_dir() else 'File'}\\n"
            info += f"Size: {stat.st_size} bytes\\n"
            info += f"Created: {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}\\n"
            info += f"Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\\n"
            info += f"Permissions: {oct(stat.st_mode)[-3:]}\\n"
            
            if file_path.is_file():
                info += f"Extension: {file_path.suffix}\\n"
                info += f"Readable: {os.access(file_path, os.R_OK)}\\n"
                info += f"Writable: {os.access(file_path, os.W_OK)}\\n"
            
            return info
        
        except Exception as e:
            return f"Error getting file info: {str(e)}"
    
    def _search_files(self, directory: str, pattern: str) -> str:
        """Search for files matching pattern."""
        try:
            dir_path = self._resolve_path(directory)
            
            if not dir_path.exists() or not dir_path.is_dir():
                return f"Error: Directory not found: {directory}"
            
            matches = []
            
            # Simple pattern matching (could be enhanced with regex)
            for file_path in dir_path.rglob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(dir_path)
                    size = file_path.stat().st_size
                    modified = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    matches.append(f"📄 {relative_path} ({size} bytes) - {modified}")
            
            if not matches:
                return f"No files found matching pattern '{pattern}' in {directory}"
            
            result = f"Search results for pattern '{pattern}' in {directory}:\\n"
            result += f"Found {len(matches)} files:\\n\\n"
            result += "\\n".join(matches)
            
            return result
        
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    async def _arun(self, command: str) -> str:
        """Async version of file manager."""
        return self._run(command)


# Example usage and testing
def test_file_manager():
    """Test the file manager tool."""
    fm = FileManagerTool()
    
    test_commands = [
        "create_dir:test_folder",
        "write_file:test_folder/sample.txt:Hello, World!\\nThis is a test file.",
        "write_file:test_folder/data.json:{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}",
        "list_dir:test_folder",
        "read_file:test_folder/sample.txt",
        "read_file:test_folder/data.json",
        "file_info:test_folder/sample.txt",
        "copy_file:test_folder/sample.txt:test_folder/sample_copy.txt",
        "search_files:test_folder:*.txt",
        "list_dir:test_folder"
    ]
    
    print("📁 File Manager Tool Test Results:")
    print("=" * 60)
    
    for cmd in test_commands:
        result = fm._run(cmd)
        print(f"Command: {cmd}")
        print(f"Result: {result}")
        print("-" * 40)


if __name__ == "__main__":
    test_file_manager()