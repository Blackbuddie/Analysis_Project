# Set Hadoop environment variables
$hadoopHome = "C:\hadoop-3.3.6"
$javaHome = "C:\Program Files\Eclipse Adoptium\jdk-11.0.27.6-hotspot"

# Update system environment variables
[System.Environment]::SetEnvironmentVariable('HADOOP_HOME', $hadoopHome, [System.EnvironmentVariableTarget]::User)
[System.Environment]::SetEnvironmentVariable('JAVA_HOME', $javaHome, [System.EnvironmentVariableTarget]::User)

# Update PATH
$path = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::User)
$newPath = ($path.Split(';') | Where-Object { $_ -notlike '*hadoop*' }) -join ';'
$newPath = "$newPath;$hadoopHome\bin"
[System.Environment]::SetEnvironmentVariable('PATH', $newPath, [System.EnvironmentVariableTarget]::User)

# Update hadoop-env.cmd
$hadoopEnvPath = "$hadoopHome\etc\hadoop\hadoop-env.cmd"
$javaHomePath = $javaHome -replace '\\', '\\'
$content = Get-Content $hadoopEnvPath -Raw
$newContent = $content -replace 'set JAVA_HOME=.*', "set JAVA_HOME=$javaHomePath"
$newContent | Set-Content $hadoopEnvPath -Force

Write-Host "Hadoop environment updated successfully!"
Write-Host "HADOOP_HOME is set to: $hadoopHome"
Write-Host "JAVA_HOME is set to: $javaHome"
Write-Host "Please restart your terminal for changes to take effect" 