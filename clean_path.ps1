# Function to clean PATH
function Clean-Path {
    param (
        [string]$pathString
    )
    
    # Split path and remove duplicates and empty entries
    $paths = $pathString.Split(';') | Where-Object { $_ -ne '' } | Select-Object -Unique
    
    # Remove Hadoop 2.8.1 related paths
    $cleanedPaths = $paths | Where-Object {
        $_ -notlike '*hadoop-2.8.1*' -and
        $_ -notlike '*%HADOOP_HOME%*' -and
        $_ -notlike '*%PATH%*'
    }
    
    return ($cleanedPaths -join ';')
}

# Clean User PATH
$userPath = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::User)
$newUserPath = Clean-Path -pathString $userPath
[System.Environment]::SetEnvironmentVariable('PATH', $newUserPath, [System.EnvironmentVariableTarget]::User)

# Clean System PATH
$systemPath = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine)
$newSystemPath = Clean-Path -pathString $systemPath
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

Write-Host "PATH has been cleaned and Hadoop 2.8.1 references removed"
Write-Host "Please restart your terminal for changes to take effect" 