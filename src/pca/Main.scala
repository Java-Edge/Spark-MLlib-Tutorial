package pca

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.sql.SparkSession

import scala.util.Random

/**
  * @author JavaEdge
  * @date 2019-04-15
  *
  */
object Main {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("iris")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN") ///日志级别

    val file = spark.read.format("csv").load("iris.data")
    //file.show()

    import spark.implicits._
    val random = new Random()
    val data = file.map(row => {
      val label = row.getString(4) match {
        case "Iris-setosa" => 0
        case "Iris-versicolor" => 1
        case "Iris-virginica" => 2
      }

      (row.getString(0).toDouble,
        row.getString(1).toDouble,
        row.getString(2).toDouble,
        row.getString(3).toDouble,
        label,
        random.nextDouble())
    }).toDF("_c0", "_c1", "_c2", "_c3", "label", "rand").sort("rand") //.where("label = 1 or label = 0")

    val assembler = new VectorAssembler().setInputCols(Array("_c0", "_c1", "_c2", "_c3")).setOutputCol("features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("features2")
      .setK(3)
    val dataset = assembler.transform(data)
    val pcaModel = pca.fit(dataset)
    val dataset2 = pcaModel.transform(dataset)
    val Array(train, test) = dataset2.randomSplit(Array(0.8, 0.2))

    val dt = new DecisionTreeClassifier().setFeaturesCol("features2").setLabelCol("label")
    val model = dt.fit(train)
    val result = model.transform(test)
    result.show(false)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(result)
    println(s"""accuracy is $accuracy""")
  }
}

