// export_per_func.sc  (JOERN 2.x + TRUE REACHING_DEF)

import java.io.PrintWriter
import scala.collection.mutable

import io.shiftleft.semanticcpg.language.*
import io.joern.dataflowengineoss.language.*
import io.joern.dataflowengineoss.layers.dataflows._
import io.joern.dataflowengineoss.layers.dataflows.OssDataFlowOptions

@main def exportGraphs(cpgFile: String, outDir: String): Unit = {

  println(s"[+] Loading CPG: $cpgFile")
  importCpg(cpgFile) 

  println("[+] Building dataflow (DDG overlay)...")
  // ⭐ THIS is the correct Joern 2.0.72 call
  run.ossdataflow
  
  val outputDir = new java.io.File(outDir)
  if (!outputDir.exists()) outputDir.mkdirs()

  println("[+] Exporting per-function graphs...")

  cpg.method
    .filterNot(_.isExternal)
    .l
    .foreach { m =>

      val nodes = mutable.Set[Long]()
      val edges = mutable.ListBuffer[(Long, Long, String)]()

      val methodNodes = m.ast.l

      // =====================================================
      // CFG
      // =====================================================
      methodNodes.foreach { src =>
        src._cfgOut.l.foreach { dst =>
          nodes += src.id
          nodes += dst.id
          edges += ((src.id, dst.id, "CFG"))
        }
      }

      // =====================================================
      // DOMINATE (CONTROL DEPENDENCE)
      // =====================================================
      methodNodes.foreach { src =>
        src._cdgOut.l.foreach { dst =>
          nodes += src.id
          nodes += dst.id
          edges += ((src.id, dst.id, "DOMINATE"))
        }
      }

      // =====================================================
      // POST_DOMINATE
      // =====================================================
      methodNodes.foreach { src =>
        src._cdgIn.l.foreach { dst =>
          nodes += src.id
          nodes += dst.id
          edges += ((src.id, dst.id, "POST_DOMINATE"))
        }
      }

      // =====================================================
      // REACHING_DEF (DDG)
      // =====================================================
      methodNodes.foreach { src =>

        val it = src.outE("REACHING_DEF")

        while (it.hasNext) {
          val e = it.next()
          val dst = e.inNode()

          nodes += src.id
          nodes += dst.id
          edges += ((src.id, dst.id, "REACHING_DEF"))
        }
      }
      // =====================================================
      // BUILD NODE MAP
      // =====================================================
      val nodeMap =
        nodes.toList
          .flatMap(id => Option(cpg.graph.node(id)).map(n => id -> n))
          .toMap

      val file =
        new java.io.File(outputDir, s"${m.name}_${m.id}.dot")

      val writer = new PrintWriter(file)

      writer.println("digraph G {")

      // ------------------ NODES ------------------
      nodeMap.foreach { case (id, n) =>

        val code =
          Option(n.property("CODE"))
            .map(_.toString.replace("\"", "'"))
            .getOrElse("")

        val nodeType =
          Option(n.label).getOrElse("UNKNOWN")

        val line =
          Option(n.property("LINE_NUMBER"))
            .map(_.toString)
            .getOrElse("-1")

        writer.println(
          s"""  $id [label="$code", code="$code", type="$nodeType", line_number="$line"];"""
        )
      }

      // ------------------ EDGES ------------------
      edges.foreach { case (src, dst, t) =>
        writer.println(s"""  $src -> $dst [type="$t"];""")
      }

      writer.println("}")
      writer.close()
    }

  println("[+] Export finished.")
}
