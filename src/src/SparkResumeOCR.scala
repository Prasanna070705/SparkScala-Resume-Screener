//> using scala "2.12"
//> using dep "org.apache.spark::spark-core:3.5.6"
//> using dep "org.apache.spark::spark-sql:3.5.6"
//> using dep "net.sourceforge.tess4j:tess4j:5.16.0"
//> using dep "org.slf4j:slf4j-simple:2.0.17"

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import net.sourceforge.tess4j.Tesseract
import java.io.File
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import scala.util.{Try, Success, Failure}
import java.nio.file.{Files, Paths}
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.PrintWriter
import java.io.OutputStreamWriter

object SparkResumeOCR_HDFS {

  case class ResumeData(filename: String, extractedText: String, status: String)

  def main(args: Array[String]): Unit = {
    println("🚀 Spark Resume OCR Processor (HDFS Integrated)")
    println("=" * 60)

    val spark = SparkSession.builder()
      .appName("Spark Resume OCR Processor (HDFS Integrated)")
      .master("local[*]") // Change to "yarn" for Hadoop cluster
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    // ✅ HDFS paths
    val inputHDFSPath = "hdfs:///user/prasanna/project_dataset/Original_dataset/"
    val outputHDFSPath = "hdfs:///user/prasanna/project_dataset/Intermediate_Dataset/resumes_extracted.csv"

    println(s"📥 Input from: $inputHDFSPath")
    println(s"💾 Output to: $outputHDFSPath")

    try {
      // Step 1: Collect image file paths from HDFS
      val fs = FileSystem.get(sc.hadoopConfiguration)
      val imageExtensions = Set(".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif")

      def listFilesRecursive(path: Path): List[String] = {
        val files = fs.listStatus(path)
        files.flatMap { file =>
          if (file.isDirectory) listFilesRecursive(file.getPath)
          else {
            val name = file.getPath.getName.toLowerCase
            if (imageExtensions.exists(name.endsWith)) Some(file.getPath.toString)
            else None
          }
        }.toList
      }

      val imageFiles = listFilesRecursive(new Path(inputHDFSPath))
      println(s"📸 Found ${imageFiles.length} image files in HDFS.")

      if (imageFiles.isEmpty) {
        println("❌ No images found in HDFS input directory.")
        return
      }

      // Step 2: OCR Processing
      def performOCR(imagePath: String): ResumeData = {
        try {
          val tesseract = new Tesseract()
          val tessDataPaths = List(
            "/usr/share/tesseract-ocr/5.00/tessdata",
            "/usr/share/tessdata",
            "/usr/local/share/tessdata"
          )

          tessDataPaths.find(path => Files.exists(Paths.get(path))) match {
            case Some(validPath) => tesseract.setDatapath(validPath)
            case None => println("⚠️ Warning: tessdata path not found, using default")
          }

          tesseract.setLanguage("eng")
          tesseract.setPageSegMode(1)
          tesseract.setOcrEngineMode(1)

          val ext = imagePath.split("\\.").lastOption.getOrElse("img")
          val localTmpFile = File.createTempFile("ocr_", s".$ext")
          fs.copyToLocalFile(new Path(imagePath), new Path(localTmpFile.getAbsolutePath))

          val image = ImageIO.read(localTmpFile)
          if (image == null) return ResumeData(imagePath.split("/").last, "", "ERROR: Invalid image")

          val text = tesseract.doOCR(image).replaceAll("[\r\n]+", " ").replaceAll("\"", "\"\"").trim
          localTmpFile.delete()

          ResumeData(imagePath.split("/").last, text, "SUCCESS")
        } catch {
          case ex: Exception =>
            ResumeData(imagePath.split("/").last, "", s"ERROR: ${ex.getMessage}")
        }
      }

      val imageRDD = sc.parallelize(imageFiles, Math.min(imageFiles.length, 8))
      val resultsRDD = imageRDD.map(performOCR)

      val results = resultsRDD.collect()
      val successful = results.count(_.status == "SUCCESS")
      val failed = results.length - successful

      println(s"✅ Successfully processed: $successful images")
      println(s"❌ Failed: $failed images")

      // Step 3: Write results to CSV on HDFS
      val csvHeader = "filename,extracted_text,status\n"
      val csvContent = new StringBuilder(csvHeader)
      results.foreach { r =>
        csvContent.append(s""""${r.filename}","${r.extractedText}","${r.status}"\n""")
      }

      val outputStream = fs.create(new Path(outputHDFSPath), true)
      val writer = new OutputStreamWriter(outputStream, "UTF-8")
      try {
        writer.write(csvContent.toString())
        println(s"💾 Successfully saved OCR results to: $outputHDFSPath")
      } finally {
        writer.close()
      }

    } catch {
      case ex: Exception =>
        println(s"❌ Error: ${ex.getMessage}")
        ex.printStackTrace()
    } finally {
      println("🏁 Closing Spark session.")
      spark.stop()
    }
  }
}
