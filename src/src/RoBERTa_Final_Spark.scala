// ==========================
// STEP 0: Import Libraries
// ==========================
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.util.Random

import ai.djl._
import ai.djl.modality.nlp._
import ai.djl.modality.nlp.bert._
import ai.djl.training._
import ai.djl.training.dataset._
import ai.djl.training.loss._
import ai.djl.training.optimizer._
import ai.djl.training.listener._
import ai.djl.inference.Predictor
import ai.djl.translate._
import ai.djl.ndarray.NDManager
import ai.djl.nn._
import ai.djl.engine.Engine
import ai.djl.training.tracker.Tracker
import java.nio.file.Paths

object RoBERTaFinalSpark {
  def main(args: Array[String]): Unit = {
    // ==========================
    // STEP 1: Initialize Spark
    // ==========================
    val spark = SparkSession.builder()
      .appName("Resume Screening with RoBERTa")
      .master("local[*]")  // Run locally using all available cores
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .config("spark.eventLog.enabled", "true")
      .config("spark.eventLog.dir", "file:///C:/temp/spark-events")
      .config("spark.history.fs.logDirectory", "file:///C:/temp/spark-events")
      .getOrCreate()
    import spark.implicits._

    // ==========================
    // STEP 2: Load Resume Dataset from HDFS
    // ==========================
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("final_removed_columns.csv")  // Read from local file

    println(s"Dataset count: ${df.count()}")
    df.show(1)

    // ==========================
    // STEP 3: Combine Sections + Completeness
    // ==========================
    val combineTextUDF = udf((education: String, experience: String, certifications: String,
                              achievements: String, languages: String, interests: String,
                              completeness: Double) => {
      val completenessLevel = completeness match {
        case x if x >= 0.7 => "high"
        case x if x >= 0.5 => "medium"
        case _ => "low"
      }
      val parts = Seq(
        Some(s"Profile completeness: $completenessLevel"),
        Option(education).filter(_.nonEmpty).map(e => s"Education: $e"),
        Option(experience).filter(_.nonEmpty).map(e => s"Experience: $e"),
        Option(certifications).filter(_.nonEmpty).map(c => s"Certifications: $c"),
        Option(achievements).filter(_.nonEmpty).map(a => s"Achievements: $a"),
        Option(languages).filter(_.nonEmpty).map(l => s"Languages: $l"),
        Option(interests).filter(_.nonEmpty).map(i => s"Interests: $i")
      ).flatten
      parts.mkString(" ")
    })

    val dfCombined = df.withColumn("combined_text",
      combineTextUDF(
        $"Education", $"Experience", $"Certifications",
        $"Achievements", $"Languages", $"Interests",
        $"completeness_score"
      )
    )

    // ==========================
    // STEP 4: Create Completeness Bins
    // ==========================
    val dfWithBins = dfCombined.withColumn("completeness_bin",
      when($"completeness_score" < 0.5, "Low")
       .when($"completeness_score" < 0.7, "Medium")
       .otherwise("High")
    )

    // ==========================
    // STEP 5: Label Encoding (Category → numeric)
    // ==========================
    import org.apache.spark.ml.feature.StringIndexer
    val labelIndexer = new StringIndexer()
      .setInputCol("Category")
      .setOutputCol("encoded_labels")
      .fit(dfWithBins)

    val dfIndexed = labelIndexer.transform(dfWithBins)
    val numLabels = dfIndexed.select("encoded_labels").distinct().count().toInt
    println(s"Number of labels: $numLabels")

    // ==========================
    // STEP 6: Create Stratification Variable
    // ==========================
    val dfStrat = dfIndexed.withColumn("strat_variable",
      concat_ws("_", $"Category", $"completeness_bin")
    )

    // Stratified Split Function
    def stratifiedSplit(df: org.apache.spark.sql.DataFrame, col: String, seed: Long, testFrac: Double):
    (org.apache.spark.sql.DataFrame, org.apache.spark.sql.DataFrame) = {
      val fractions = df.select(col).distinct().as[String].collect().map(c => c -> testFrac).toMap
      val test = df.stat.sampleBy(col, fractions, seed)
      val train = df.except(test)
      (train, test)
    }

    // 80/10/10 split
    val (trainDF, valTestDF) = stratifiedSplit(dfStrat, "strat_variable", 42L, 0.2)
    val (valDF, testDF) = stratifiedSplit(valTestDF, "strat_variable", 42L, 0.5)

    println(s"Train: ${trainDF.count()}, Val: ${valDF.count()}, Test: ${testDF.count()}")

    // ==========================
    // STEP 7: Tokenization using DJL + RoBERTa
    // ==========================
    // Use BertTokenizer without arguments, or use HuggingFaceTokenizer if available.
    val tokenizer = new BertTokenizer()

    // Convert Spark DF to DJL Dataset
    def createDJLDataset(df: org.apache.spark.sql.DataFrame, textCol: String, labelCol: String) = {
      val data = df.select(textCol, labelCol).collect().map { row =>
        val text = row.getString(0)
        val label = row.getDouble(1).toInt
        (text, label)
      }
      data
    }

    val trainDataset = createDJLDataset(trainDF, "combined_text", "encoded_labels")
    val valDataset = createDJLDataset(valDF, "combined_text", "encoded_labels")
    val testDataset = createDJLDataset(testDF, "combined_text", "encoded_labels")

    // ==========================
    // STEP 8: Model Setup (COMMENTED OUT FOR NOW - FOCUS ON SPARK PROCESSING)
    // ==========================
    println("=== SPARK DATA PROCESSING COMPLETED SUCCESSFULLY ===")
    println(s"Training data count: ${trainDataset.length}")
    println(s"Validation data count: ${valDataset.length}")
    println(s"Test data count: ${testDF.count()}")
    
    // Skip DJL model training for now - uncomment when engine is properly configured
    /*
    import ai.djl.nn.core.Linear
    import ai.djl.ndarray.types.Shape

    val model = ai.djl.Model.newInstance("roberta-base")
    val block = new SequentialBlock()
    block.add(Linear.builder().setUnits(numLabels).build())
    model.setBlock(block)

    val loss = Loss.softmaxCrossEntropyLoss()
    val optimizer = Optimizer.adam().build()
    val config = new DefaultTrainingConfig(loss)
      .optOptimizer(optimizer)
      .addEvaluator(new ai.djl.training.evaluator.Accuracy())
      .addTrainingListeners(TrainingListener.Defaults.logging().toArray: _*)

    val trainer = model.newTrainer(config)
    trainer.initialize(new Shape(1, 512)) // batch_size x seq_len

    // ==========================
    // STEP 9: Training Loop
    // ==========================
    for (epoch <- 1.to(5)) {
      println(s"Epoch $epoch/5")
      trainDataset.foreach { case (text, label) =>
        // Implement DJL training logic here
        // See DJL docs for correct usage
      }
      valDataset.foreach { case (text, label) =>
        // Implement DJL validation logic here
      }
      // Notify listeners if needed
    }

    // ==========================
    // STEP 10: Evaluation
    // ==========================
    import ai.djl.translate.Translator
    import ai.djl.translate.TranslatorContext

    class SimpleTranslator extends Translator[String, Int] {
      override def processInput(ctx: TranslatorContext, input: String): ai.djl.ndarray.NDList = {
        // Tokenize input and convert to NDList
        ???
      }
      override def processOutput(ctx: TranslatorContext, list: ai.djl.ndarray.NDList): Int = {
        // Convert output NDList to label
        ???
      }
    }

    val predictor = model.newPredictor(new SimpleTranslator())

    val testTexts = testDF.select("combined_text").as[String].collect()
    val testLabels = testDF.select("encoded_labels").as[Double].map(_.toInt).collect()
    val predLabels = testTexts.map(predictor.predict)

    val correct = predLabels.zip(testLabels).count { case (pred, label) => pred == label }
    val accuracy = correct.toDouble / testLabels.length
    println(s"Accuracy: $accuracy")
    */
    
    println("=== APPLICATION COMPLETED SUCCESSFULLY ===")
    println("Note: DJL model training is commented out. Uncomment after configuring DJL engine.")
    
    // Save Spark-processed data locally since we're using local mode
    println("Saving processed data...")
    trainDF.write.mode("overwrite").parquet("output/train_data")
    valDF.write.mode("overwrite").parquet("output/val_data") 
    testDF.write.mode("overwrite").parquet("output/test_data")
    
    println("Data processing pipeline completed successfully!")
    println("Files saved in output/ directory:")
    println("- output/train_data/")
    println("- output/val_data/") 
    println("- output/test_data/")
    
    /*
    // STEP 9: Training Loop
    // ==========================
    for (epoch <- 1.to(5)) {
      println(s"Epoch $epoch/5")
      trainDataset.foreach { case (text, label) =>
        // Implement DJL training logic here
        // See DJL docs for correct usage
      }
      valDataset.foreach { case (text, label) =>
        // Implement DJL validation logic here
      }
      // Notify listeners if needed
    }

    // ==========================
    // STEP 10: Evaluation
    // ==========================
    import ai.djl.translate.Translator
    import ai.djl.translate.TranslatorContext

    class SimpleTranslator extends Translator[String, Int] {
      override def processInput(ctx: TranslatorContext, input: String): ai.djl.ndarray.NDList = {
        // Tokenize input and convert to NDList
        ???
      }
      override def processOutput(ctx: TranslatorContext, list: ai.djl.ndarray.NDList): Int = {
        // Convert output NDList to label
        ???
      }
    }

    val predictor = model.newPredictor(new SimpleTranslator())

    val testTexts = testDF.select("combined_text").as[String].collect()
    val testLabels = testDF.select("encoded_labels").as[Double].map(_.toInt).collect()
    val predLabels = testTexts.map(predictor.predict)

    val correct = predLabels.zip(testLabels).count { case (pred, label) => pred == label }
    val accuracy = correct.toDouble / testLabels.length
    println(s"Test Accuracy: $accuracy")

    // ==========================
    // STEP 11: Save Model + Artifacts to HDFS
    // ==========================
    val hdfsPath = "hdfs:///user/prasanna/project_dataset/Results_after_FineTuning"
    model.save(Paths.get("/tmp/roberta_resume_model"), "roberta")
    labelIndexer.write.overwrite().save("/tmp/label_encoder")

    import org.apache.hadoop.fs.{FileSystem, Path}
    import org.apache.hadoop.conf.Configuration

    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    fs.copyFromLocalFile(new Path("/tmp/roberta_resume_model"), new Path(hdfsPath + "/roberta_model"))
    fs.copyFromLocalFile(new Path("/tmp/label_encoder"), new Path(hdfsPath + "/label_encoder"))

    println(s"✅ Training completed and artifacts saved to HDFS: $hdfsPath")

    // ==========================
    // STEP 12: Completeness Strategy Recommendation
    // ==========================
    println("="*50)
    println("COMPLETENESS STRATEGY EVALUATION")
    println("="*50)
    println("Monitor the performance by completeness levels above.")
    println("If low-completeness resumes show significantly worse performance,")
    println("consider running the model again with filtering (completeness_score >= 0.5)")
    println("="*50)
    */
    
    spark.stop()
  }

  def createDJLDataset(df: org.apache.spark.sql.DataFrame, textCol: String, labelCol: String): Array[(String, Int)] = {
    df.select(textCol, labelCol).collect().map(row => (row.getString(0), row.getInt(1)))
  }
}
