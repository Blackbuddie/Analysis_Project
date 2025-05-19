param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('8', '11')]
    [string]$Version
)

if ($Version -eq '8') {
    $javaHome = "C:\Program Files\Java\jdk1.8.0_202"
} else {
    $javaHome = "C:\Program Files\Eclipse Adoptium\jdk-11.0.27.6-hotspot"
}

# Set JAVA_HOME
[System.Environment]::SetEnvironmentVariable('JAVA_HOME', $javaHome, [System.EnvironmentVariableTarget]::User)

# Update PATH
$path = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::User)
$newPath = ($path.Split(';') | Where-Object { $_ -notlike '*Java*' }) -join ';'
$newPath = "$newPath;$javaHome\bin"
[System.Environment]::SetEnvironmentVariable('PATH', $newPath, [System.EnvironmentVariableTarget]::User)

Write-Host "Switched to Java $Version"
Write-Host "JAVA_HOME is now set to: $javaHome"
Write-Host "Please restart your terminal for changes to take effect" 