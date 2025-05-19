# Set environment variables
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

# Create core-site.xml if it doesn't exist
$coreSitePath = "$hadoopHome\etc\hadoop\core-site.xml"
if (-not (Test-Path $coreSitePath)) {
    @"
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
"@ | Set-Content $coreSitePath
}

# Create hdfs-site.xml if it doesn't exist
$hdfsSitePath = "$hadoopHome\etc\hadoop\hdfs-site.xml"
if (-not (Test-Path $hdfsSitePath)) {
    @"
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>C:/hadoop-3.3.6/data/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>C:/hadoop-3.3.6/data/datanode</value>
    </property>
</configuration>
"@ | Set-Content $hdfsSitePath
}

# Create mapred-site.xml if it doesn't exist
$mapredSitePath = "$hadoopHome\etc\hadoop\mapred-site.xml"
if (-not (Test-Path $mapredSitePath)) {
    @"
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
"@ | Set-Content $mapredSitePath
}

# Create yarn-site.xml if it doesn't exist
$yarnSitePath = "$hadoopHome\etc\hadoop\yarn-site.xml"
if (-not (Test-Path $yarnSitePath)) {
    @"
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
</configuration>
"@ | Set-Content $yarnSitePath
}

# Update hadoop-env.cmd
$hadoopEnvPath = "$hadoopHome\etc\hadoop\hadoop-env.cmd"
$javaHomePath = $javaHome -replace '\\', '\\'
$content = Get-Content $hadoopEnvPath -Raw
$newContent = $content -replace 'set JAVA_HOME=.*', "set JAVA_HOME=$javaHomePath"
$newContent | Set-Content $hadoopEnvPath -Force

# Create data directories
$dataDirs = @(
    "$hadoopHome\data\namenode",
    "$hadoopHome\data\datanode"
)

foreach ($dir in $dataDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
    }
}

Write-Host "Hadoop setup completed successfully!"
Write-Host "HADOOP_HOME is set to: $hadoopHome"
Write-Host "JAVA_HOME is set to: $javaHome"
Write-Host "Please restart your terminal for changes to take effect" 