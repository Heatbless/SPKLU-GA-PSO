Set fileSystem = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
projectFolder = fileSystem.GetParentFolderName(WScript.ScriptFullName)
pythonWindow = projectFolder & "\.venv-desktop\Scripts\pythonw.exe"
applicationFile = projectFolder & "\app.py"
If Not fileSystem.FileExists(pythonWindow) Then
  MsgBox "Install the app environment first by following the Quick start in README.md.",  vbExclamation, "SPKLU Site Optimizer"
Else
  shell.Run Chr(34) & pythonWindow & Chr(34) & " " & Chr(34) & applicationFile & Chr(34), 0, False
End If
