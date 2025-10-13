// ==========================
// STEP 0: Import Libraries
// ==========================
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{Tokenizer, NerDLModel, NerConverter}
import com.johnsnowlabs.nlp.Finisher
import org.apache.spark.ml.Pipeline

object SparkNERDataCleaning {
  def main(args: Array[String]): Unit = {
    // ==========================
    // STEP 1: Initialize Spark
    // ==========================
    val spark = SparkSession.builder()
      .appName("Resume NER + ESCO Extraction")
      .getOrCreate()
    import spark.implicits._

    // ==========================
    // STEP 2: Load Resume Dataset
    // ==========================
    val resumeDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("hdfs:///user/prasanna/project_dataset/Intermediate_Dataset/resumes_extracted.csv")

    println(s"Resume dataset count: ${resumeDF.count()}")
    resumeDF.show(1)

    // ==========================
    // STEP 3: Load ESCO Skills Dataset
    // ==========================
    val escoDF = spark.read
      .option("header", "true")
      .csv("hdfs:///user/prasanna/project_dataset/ESCO_Dataset/skills_en")

    val escoSkills = escoDF.select("preferredLabel")
      .na.drop()
      .as[String]
      .map(_.toLowerCase.trim)
      .collect()
      .toSet

    println(s"Total ESCO skills loaded: ${escoSkills.size}")

    // ==========================
    // STEP 4: SparkNLP NER Model
    // ==========================
    // Load pre-trained resume NER model
    val nerModel = NerDLModel.pretrained("resume_ner", "en").setInputCols("token", "document").setOutputCol("ner")
    val documentAssembler = new DocumentAssembler().setInputCol("Text").setOutputCol("document")
    val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
    val nerConverter = new NerConverter().setInputCols("document", "token", "ner").setOutputCol("ner_chunks")
    val finisher = new Finisher().setInputCols("ner_chunks").setOutputCols("finished_ner").setCleanAnnotations(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, nerModel, nerConverter, finisher))

    // Fit pipeline (transforming the dataset)
    val pipelineModel = pipeline.fit(resumeDF)
    val nerDF = pipelineModel.transform(resumeDF)

    // ==========================
    // STEP 5: Extract Skills
    // ==========================
    val extractSkillsUDF = udf { nerList: Seq[String] =>
      if (nerList != null) {
        val nerSkills = nerList.map(_.toLowerCase)
        val matchedESCO = escoSkills.filter(skill => nerSkills.exists(_.contains(skill)))
        (nerSkills ++ matchedESCO).distinct
      } else Seq.empty[String]
    }

    val finalDF = nerDF.withColumn("extracted_skills", extractSkillsUDF($"finished_ner"))

    // ==========================
    // STEP 6: Section Extraction Functions
    // ==========================
    def extractSection(text: String, sectionName: String): String = {
      if (text == null) return null
      val upperText = text.toUpperCase
      val sectionHeader = sectionName.toUpperCase
      val pattern = (sectionHeader + """[\s\S]*?(?=\n[A-Z ]+\n|$)""").r
      pattern.findFirstIn(upperText) match {
        case Some(matched) => matched.replaceFirst(sectionHeader, "").trim
        case None => null
      }
    }

    val extractSectionUDF = udf((text: String, sectionName: String) => extractSection(text, sectionName))

    val sectionsDF = finalDF
      .withColumn("Education", extractSectionUDF($"Text", lit("EDUCATION")))
      .withColumn("Certifications", coalesce(
        extractSectionUDF($"Text", lit("CERTIFICATIONS")),
        extractSectionUDF($"Text", lit("TRAINING & CERTIFICATIONS"))
      ))
      .withColumn("Achievements", extractSectionUDF($"Text", lit("ACHIEVEMENTS")))
      .withColumn("Experience", coalesce(
        extractSectionUDF($"Text", lit("EXPERIENCE")),
        extractSectionUDF($"Text", lit("PROFESSIONAL EXPERIENCE"))
      ))
      .withColumn("Languages", extractSectionUDF($"Text", lit("LANGUAGES")))
      .withColumn("Interests", extractSectionUDF($"Text", lit("INTERESTS")))

    // ==========================
    // STEP 7: Save Output
    // ==========================
    sectionsDF.write
      .option("header", "true")
      .mode("overwrite")
      .csv("hdfs:///user/prasanna/project_dataset/Cleaned_Dataset_for_BERT/final_removed_columns.csv")

    println("✅ Skills and sections extracted and saved successfully!")
  }
}
