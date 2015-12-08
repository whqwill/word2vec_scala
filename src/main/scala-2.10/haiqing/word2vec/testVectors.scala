package haiqing.word2vec

/**
 * Created by hwang on 07.12.15.
 */
object testVectors {
  def main(args: Array[String]): Unit = {
    val model = Processing.loadModel("./data_from_cluster")

    val synonyms = model.findSynonyms("apple", 20)
    //val synonyms = model.findSynonyms("day", 10)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    println()
  }

}
