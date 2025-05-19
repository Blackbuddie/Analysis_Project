# Remove Hadoop 2.8.1 from PATH in User variables
$userPath = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::User)
$newUserPath = ($userPath.Split(';') | Where-Object { 
    $_ -notlike '*hadoop-2.8.1*' -and 
    $_ -notlike '*hadoop-2.8.1\bin*' -and
    $_ -notlike '*hadoop-2.8.1' 
}) -join ';'
[System.Environment]::SetEnvironmentVariable('PATH', $newUserPath, [System.EnvironmentVariableTarget]::User)

# Remove Hadoop 2.8.1 from PATH in System variables
$systemPath = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine)
$newSystemPath = ($systemPath.Split(';') | Where-Object { 
    $_ -notlike '*hadoop-2.8.1*' -and 
    $_ -notlike '*hadoop-2.8.1\bin*' -and
    $_ -notlike '*hadoop-2.8.1' 
}) -join ';'
[System.Environment]::SetEnvironmentVariable('PATH', $newSystemPath, [System.EnvironmentVariableTarget]::Machine)

# Remove HADOOP_HOME if it points to 2.8.1
$hadoopHome = [System.Environment]::GetEnvironmentVariable('HADOOP_HOME', [System.EnvironmentVariableTarget]::User)
if ($hadoopHome -like '*hadoop-2.8.1*') {
    [System.Environment]::SetEnvironmentVariable('HADOOP_HOME', $null, [System.EnvironmentVariableTarget]::User)
}

# Also check and remove from Machine level
$hadoopHomeMachine = [System.Environment]::GetEnvironmentVariable('HADOOP_HOME', [System.EnvironmentVariableTarget]::Machine)
if ($hadoopHomeMachine -like '*hadoop-2.8.1*') {
    [System.Environment]::SetEnvironmentVariable('HADOOP_HOME', $null, [System.EnvironmentVariableTarget]::Machine)
}

Write-Host "Hadoop 2.8.1 has been removed from environment variables"
Write-Host "Please restart your terminal for changes to take effect" 