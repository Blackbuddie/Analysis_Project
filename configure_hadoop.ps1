# Set environment variables
$hadoopHome = "C:\hadoop-3.3.6"
$javaHome = "C:\Program Files\Eclipse Adoptium\jdk-11.0.27.6-hotspot"

# Update hadoop-env.cmd
$hadoopEnvPath = "$hadoopHome\etc\hadoop\hadoop-env.cmd"
$javaHomePath = $javaHome -replace '\\', '\\'

# Create a simpler hadoop-env.cmd
$newContent = @"
@rem Licensed to the Apache Software Foundation (ASF) under one
@rem or more contributor license agreements.  See the NOTICE file
@rem distributed with this work for additional information
@rem regarding copyright ownership.  The ASF licenses this file
@rem to you under the Apache License, Version 2.0 (the
@rem "License"); you may not use this file except in compliance
@rem with the License.  You may obtain a copy of the License at
@rem
@rem     http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing,
@rem software distributed under the License is distributed on an
@rem "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
@rem KIND, either express or implied.  See the License for the
@rem specific language governing permissions and limitations
@rem under the License.

@rem Set Hadoop-specific environment variables here.

@rem The java implementation to use.  Required.
set JAVA_HOME=$javaHomePath

@rem The maximum amount of heap to use, in MB. Default is 1000.
set HADOOP_HEAPSIZE_MAX=1000

@rem The minimum amount of heap to use, in MB. Default is 1000.
set HADOOP_HEAPSIZE_MIN=1000

@rem Extra Java runtime options.
set HADOOP_OPTS=-Djava.net.preferIPv4Stack=true

@rem Command specific options appended to HADOOP_OPTS when specified
set HADOOP_NAMENODE_OPTS=-Xmx1024m
set HADOOP_DATANODE_OPTS=-Xmx1024m
"@

# Write the new content to hadoop-env.cmd
$newContent | Set-Content $hadoopEnvPath -Force

# Also create a hadoop-env.sh for Unix-style commands
$hadoopEnvShPath = "$hadoopHome\etc\hadoop\hadoop-env.sh"
$newContentSh = $newContent -replace '@rem', '#' -replace 'set ', 'export '
$newContentSh | Set-Content $hadoopEnvShPath -Force

Write-Host "Hadoop environment has been configured successfully!"
Write-Host "HADOOP_HOME is set to: $hadoopHome"
Write-Host "JAVA_HOME is set to: $javaHome"
Write-Host "Please restart your terminal for changes to take effect" 