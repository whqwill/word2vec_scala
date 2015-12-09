package haiqing.word2vec

/**
 * Created by hwang on 07.12.15.
 */
object testVectors {
  def main(args: Array[String]): Unit = {
    //val model = Processing.loadModel("./data_summing_version")
    val model = Processing.loadTmpModel("./data_summing_version",6000)

    val synonyms = model.findSynonyms("say", 20)
    //val synonyms = model.findSynonyms("day", 10)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    println()
    println()
    val NEWsynonyms = model.findSynonyms("bank", 20)
    //val synonyms = model.findSynonyms("day", 10)

    for((synonym, cosineSimilarity) <- NEWsynonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    println()
  }

}
