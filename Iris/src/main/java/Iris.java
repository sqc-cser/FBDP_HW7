import java.awt.datatransfer.StringSelection;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;


public class Iris {

    public static class TokenizerMapper
            extends Mapper<Object, Text, IntWritable,Text >{
        // 存放测试集路径
        private String localFiles;
        // 存放测试数据
        private List test = new ArrayList();
        @Override
        public void setup(Context context) throws IOException,InterruptedException{
            Configuration conf = context.getConfiguration();
            // 获取测试集所在的hdfs路径
            localFiles  = conf.getStrings("test")[0];
            FileSystem fs = FileSystem.get(URI.create(localFiles), conf);
            FSDataInputStream hdfsInStream = fs.open(new Path(localFiles));
            // 从hdfs中读取测试集
            InputStreamReader isr = new InputStreamReader(hdfsInStream, "utf-8");
            String line;
            BufferedReader br = new BufferedReader(isr);
            while ((line = br.readLine()) != null) {
                StringTokenizer itr = new StringTokenizer(line);
                while (itr.hasMoreTokens()) {
                    String[] tmp = itr.nextToken().split(",");
                    List data = new ArrayList();
                    for (String i : tmp){
                        data.add(Double.parseDouble(i));
                    }
                    test.add(data);
                }
            }
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                // 将训练数据分割
                String[] tmp = itr.nextToken().split(",");
                // 记录该训练集的标签
                String label = tmp[4];
                // 记录该训练集的属性值
                List data = new ArrayList();
                for (int i = 0;i<=3;i++){
                    data.add(Double.parseDouble(tmp[i]));
                }
                for (int i = 0;i<test.size();i++){
                    // 获得每个测试数据
                    List tmp2 = (List) test.get(i);
                    // 每个测试数据和训练数据的距离(这里使用欧氏距离)
                    double dis = 0;
                    for (int j=0;j<4;j++){
                        dis += Math.pow( (double)tmp2.get(j)-(double)data.get(j),2);
                    }
                    dis = Math.sqrt(dis);
                    // out 为类标签,距离
                    String out = label + "," + String.valueOf(dis);
                    context.write(new IntWritable(i), new Text(out));
                }
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<IntWritable,Text,IntWritable,Text> {

        private String localFiles;
        private List tgt = new ArrayList();
        private int n;
        //读取测试集的标签
        @Override
        public void setup(Context context) throws IOException,InterruptedException{
            Configuration conf = context.getConfiguration();
            // 获取测试集标签所在的hdfs路径
            localFiles  = conf.getStrings("label")[0];
            // 读取n值
            n = conf.getInt("n", 3);
            FileSystem fs = FileSystem.get(URI.create(localFiles), conf);
            FSDataInputStream hdfsInStream = fs.open(new Path(localFiles));
            // 从hdfs中读取测试集
            InputStreamReader isr = new InputStreamReader(hdfsInStream, "utf-8");
            String line;
            BufferedReader br = new BufferedReader(isr);
            while ((line = br.readLine()) != null) {
                StringTokenizer itr = new StringTokenizer(line);
                while (itr.hasMoreTokens()) {
                    tgt.add(itr.nextToken());
                }
            }
        }

        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            List<String> sortvalue = new ArrayList<String>();
            for (Text val : values) {
                sortvalue.add(val.toString());
            }

            // 对距离进行排序
            Collections.sort(sortvalue, new Comparator<String>() {

                @Override
                public int compare(String o1, String o2) {
                    // 升序
                    double x = Double.parseDouble(o1.split(",")[1]);
                    double y = Double.parseDouble(o2.split(",")[1]);
                    return Double.compare(x, y);
                }
            });
            List<String> labels = new ArrayList<String>();
            for (int i =0;i<n;i++){
                labels.add(sortvalue.get(i).split(",")[0]);
            }
            // 将标签转换成集合方便计数
            Set<String> set = new LinkedHashSet<>();
            set.addAll(labels);
            List<String> labelset = new ArrayList<>(set);
            int[] count = new int[labelset.size()];
            // 将计数数组全部初始化为0
            for (int i=0;i<count.length;i++){
                count[i] = 0;
            }
            // 对每个标签计数得到count，位置对应labelset
            for(int i=0;i<labelset.size();i++){
                for (int j=0;j<labels.size();j++){
                    if (labelset.get(i).equals(labels.get(j))){
                        count[i] += 1;
                    }
                }
            }

            // 求count最大值所在的索引
            int max = 0;
            for(int i=1;i<count.length;i++){
                if(count[i] > count[max]){
                    max = i;
                }
            }
            context.write(key, new Text("预测标签:" + labelset.get(max) + "\t" + "真实标签:" + String.valueOf(tgt.get(key.get()))));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        conf.setStrings("label", otherArgs[1]);
        conf.setStrings("test", otherArgs[2]);
        // 从命令行传入参数N
        conf.setInt("n", Integer.parseInt(otherArgs[0]));
        //传参格式为k，测试集对应的标签，测试集，训练集，输出路径
        if (otherArgs.length < 5) {
            System.err.println("输入格式: <k> <label> <test> <train> <output>");
            System.exit(-1);
        }
        Job job = Job.getInstance(conf);
        job.setJarByClass(Iris.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        for (int i = 3; i < otherArgs.length - 1; ++i) { // 训练文件可以传入多个
            //由于是训练集大的情况，所以将训练集输入
            FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
        }
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[otherArgs.length - 1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

