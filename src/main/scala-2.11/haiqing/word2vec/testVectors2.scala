package haiqing.word2vec

/**
 * Created by hwang on 26.01.16.
 */
object testVectors2 {
  def main(args: Array[String]): Unit = {
    val model = Processing.loadModelSenses(args(0))
    //val model = Processing.loadTmpModel("./data_new",2000)

    val synonyms = model.findSynonyms(args(1), 20)
    //val synonyms = model.findSynonyms("day", 10)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"${synonym.substring(0,synonym.size-2)} $cosineSimilarity")
    }

    println()
    println()
    val NEWsynonyms = model.findSynonyms(args(2), 20)
    //val synonyms = model.findSynonyms("day", 10)

    for((synonym, cosineSimilarity) <- NEWsynonyms) {
      println(s"${synonym.substring(0,synonym.size-2)} $cosineSimilarity")
    }
    println()




  }
}
