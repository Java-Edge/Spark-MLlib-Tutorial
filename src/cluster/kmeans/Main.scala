package cluster.kmeans

import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

import scala.util.Random

/**
  * @author JavaEdge
  * @date 2019-04-17
  *
  */
object Main {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("iris")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    val file = spark.read.format("csv").load("iris.data")
    file.show()

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
    }).toDF("_c0", "_c1", "_c2", "_c3", "label", "rand").sort("rand")

    val assembler = new VectorAssembler()
      .setInputCols(Array("_c0", "_c1", "_c2", "_c3"))
      .setOutputCol("features")

    // 分割
    val dataset = assembler.transform(data)
    val Array(train, test) = dataset.randomSplit(Array(0.8, 0.2))
    train.show()

    // kmeans 算法
    val kmeans = new KMeans().setFeaturesCol("features").setK(3).setMaxIter(20)
    val model = kmeans.fit(train)

    model.transform(train).show()

  }
}
