package isotonic

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.IsotonicRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

/**
  * @author JavaEdge
  * @date 2019-04-15
  *
  */
object Main {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("linear").setMaster("local")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().config(conf).getOrCreate()

    val file = spark.read.format("csv").option("sep", ";").option("header", "true").load("house.csv")
    import spark.implicits._
    //打乱顺序
    val rand = new Random()
    val data = file.select("square", "price").map(
      row => (row.getAs[String](0).toDouble, row.getString(1).toDouble, rand.nextDouble()))
      .toDF("square", "price", "rand").sort("rand") //强制类型转换过程

    val ass = new VectorAssembler().setInputCols(Array("square")).setOutputCol("features")
    val dataset = ass.transform(data) //特征包装
    val Array(train, test) = dataset.randomSplit(Array(0.8, 0.2)) //拆分成训练数据集和测试数据集

    val isotonic = new IsotonicRegression().setFeaturesCol("features").setLabelCol("price")
    val model = isotonic.fit(train)
    model.transform(test).show()
  }

}

