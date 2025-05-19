# Download Hadoop 3.3.6
$hadoopVersion = "3.3.6"
$hadoopUrl = "https://dlcdn.apache.org/hadoop/common/hadoop-$hadoopVersion/hadoop-$hadoopVersion.tar.gz"
$downloadPath = "$env:USERPROFILE\Downloads\hadoop-$hadoopVersion.tar.gz"
$installPath = "C:\hadoop-$hadoopVersion"

# Create installation directory if it doesn't exist
if (-not (Test-Path $installPath)) {
    New-Item -ItemType Directory -Path $installPath
}

# Download Hadoop
Write-Host "Downloading Hadoop $hadoopVersion..."
Invoke-WebRequest -Uri $hadoopUrl -OutFile $downloadPath

# Extract the archive
Write-Host "Extracting Hadoop..."
tar -xf $downloadPath -C "C:\"

# Set environment variables
[System.Environment]::SetEnvironmentVariable('HADOOP_HOME', $installPath, [System.EnvironmentVariableTarget]::User)
[System.Environment]::SetEnvironmentVariable('PATH', $env:PATH + ";$installPath\bin", [System.EnvironmentVariableTarget]::User)

# Update hadoop-env.cmd
$hadoopEnvPath = "$installPath\etc\hadoop\hadoop-env.cmd"
$javaHome = "C:\Program Files\Java\jdk-11"
$javaHomePath = $javaHome -replace '\\', '\\'

# Read and update the content
$content = Get-Content $hadoopEnvPath -Raw
$newContent = $content -replace 'set JAVA_HOME=.*', "set JAVA_HOME=$javaHomePath"
$newContent | Set-Content $hadoopEnvPath -Force

Write-Host "Hadoop $hadoopVersion has been installed successfully!"
Write-Host "Please restart your terminal for the changes to take effect." 