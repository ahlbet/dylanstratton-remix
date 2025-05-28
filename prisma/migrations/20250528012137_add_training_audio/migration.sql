-- CreateTable
CREATE TABLE "TrainingAudio" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "filename" TEXT NOT NULL,
    "objectKey" TEXT NOT NULL,
    "duration" REAL NOT NULL,
    "fileSize" INTEGER NOT NULL,
    "mimeType" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "userId" TEXT NOT NULL,
    CONSTRAINT "TrainingAudio_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex
CREATE INDEX "TrainingAudio_userId_idx" ON "TrainingAudio"("userId");

-- CreateIndex
CREATE INDEX "TrainingAudio_userId_createdAt_idx" ON "TrainingAudio"("userId", "createdAt");
