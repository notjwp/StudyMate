"use client";

import { UploadCloud } from "lucide-react";
import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "./ui/button";

interface FileUploadProps {
  onFileChange: (file: File | null) => void;
  acceptedTypes?: { [key: string]: string[] };
  disabled?: boolean;
}

export default function FileUpload({
  onFileChange,
  acceptedTypes = { "application/pdf": [".pdf"] },
  disabled = false,
}: FileUploadProps) {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setUploadedFile(file);
        onFileChange(file);
      }
    },
    [onFileChange]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedTypes,
    maxFiles: 1,
    disabled,
  });

  const handleRemoveFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setUploadedFile(null);
    onFileChange(null);
  };

  return (
    <div
      {...getRootProps()}
      className={`flex flex-col items-center justify-center w-full p-6 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
        isDragActive
          ? "border-primary bg-primary/10"
          : "border-border hover:border-primary/50"
      } ${disabled ? "cursor-not-allowed opacity-50 bg-muted/50" : "bg-background"}`}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center">
        <UploadCloud
          className={`w-10 h-10 mb-3 ${
            isDragActive ? "text-primary" : "text-gray-400"
          }`}
        />
        {uploadedFile ? (
          <div>
            <p className="font-semibold">{uploadedFile.name}</p>
            <p className="text-xs text-muted-foreground">
              {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
            <Button
              onClick={handleRemoveFile}
              variant="link"
              size="sm"
              className="mt-2 text-destructive hover:underline"
            >
              Remove file
            </Button>
          </div>
        ) : (
          <div>
            <p className="mb-2 text-sm text-muted-foreground">
              <span className="font-semibold text-primary">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-muted-foreground">PDF (MAX. 10MB)</p>
          </div>
        )}
      </div>
    </div>
  );
}
