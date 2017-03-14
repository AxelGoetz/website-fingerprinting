package main

import (
	"bufio"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// FeatureDelimiter is the delimiter in the output between features
const FeatureDelimiter = " "

func extract(times []float64, sizes []int) (features string, err error) {
	// transmission size features
	count := 0
	for _, s := range sizes {
		if s > 0 {
			count++
		}
	}
	features = strconv.Itoa(len(times))
	features += FeatureDelimiter + strconv.Itoa(count)
	features += FeatureDelimiter + strconv.Itoa(len(times)-count)
	features += FeatureDelimiter + strconv.FormatFloat((times[len(times)-1]-times[0]), 'f', -1, 64)

	// unique packet lengts
	for i := -1500; i < 1501; i++ {
		in := false
		for _, s := range sizes {
			if s == i {
				in = true
				break
			}
		}
		if in {
			features += FeatureDelimiter + strconv.Itoa(1)
		} else {
			features += FeatureDelimiter + strconv.Itoa(0)
		}
	}

	// transpositions (similar to good distance scheme)
	count = 0
	for i := 0; i < len(sizes); i++ {
		if sizes[i] > 0 {
			count++
			features += FeatureDelimiter + strconv.Itoa(i)
		}

		if count == 300 {
			break
		}
	}
	for i := count; i < 300; i++ {
		features += FeatureDelimiter + "'X'"
	}

	count = 0
	prevloc := 0
	for i := 0; i < len(sizes); i++ {
		if sizes[i] > 0 {
			count++
			features += FeatureDelimiter + strconv.Itoa(i-prevloc)
			prevloc = i
		}
		if count == 300 {
			break
		}
	}
	for i := count; i < 300; i++ {
		features += FeatureDelimiter + "'X'"
	}

	// packet distributions (where are the outgoing packets concentrated)
	// TODO: missing count = 0 reset here, but porting bug for now
	// count = 0
	for i := 0; i < len(sizes) && i < 3000; i++ {
		if i%30 != 29 {
			if sizes[i] > 0 {
				count++
			}
		} else {
			features += FeatureDelimiter + strconv.Itoa(count)
			count = 0
		}
	}
	for i := len(sizes) / 30; i < 100; i++ {
		features += FeatureDelimiter + strconv.Itoa(0)
	}

	// Bursts
	curburst := 0
	stopped := false
	var bursts []int
	for i := 0; i < len(sizes); i++ {
		if sizes[i] < 0 {
			stopped = false
			curburst -= sizes[i]
		}
		if sizes[i] > 0 && !stopped {
			stopped = true
		}
		if sizes[i] > 0 && stopped {
			stopped = false
			bursts = append(bursts, curburst)
		}
	}
	max := -1
	sum := 0
	for i := 0; i < len(bursts); i++ {
		sum += bursts[i]
		if bursts[i] > max {
			max = bursts[i]
		}
	}
	features += FeatureDelimiter + strconv.Itoa(max)
	features += FeatureDelimiter + strconv.Itoa(sum/len(bursts))
	features += FeatureDelimiter + strconv.Itoa(len(bursts))

	counts := make([]int, 3)
	for i := 0; i < len(bursts); i++ {
		if bursts[i] > 5 {
			counts[0]++
		}
		if bursts[i] > 10 {
			counts[1]++
		}
		if bursts[i] > 15 {
			counts[2]++
		}
	}
	features += FeatureDelimiter + strconv.Itoa(counts[0])
	features += FeatureDelimiter + strconv.Itoa(counts[1])
	features += FeatureDelimiter + strconv.Itoa(counts[2])

	for i := 0; i < 5; i++ {
		if len(bursts) > i {
			features += FeatureDelimiter + strconv.Itoa(bursts[i])
		} else {
			features += FeatureDelimiter + "'X'"
		}
	}

	for i := 0; i < 20; i++ {
		if len(sizes) > i {
			features += FeatureDelimiter + strconv.Itoa(sizes[i]+1500)
		} else {
			features += FeatureDelimiter + "'X'"
		}
	}

	return
}

func parse(filename, suffix string, new_path string) {
	file, err := os.Open(filename + ".cell")
	if err != nil {
		log.Fatalf("failed to read file %s, got error %s", filename, err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	var times []float64
	var sizes []int
	for scanner.Scan() {
		items := strings.Split(scanner.Text(), "\t")
		if len(items) != 2 {
			log.Fatalf("expected 2 items in line for filename %s, got %d", filename, len(items))
		}

		t, er := strconv.ParseFloat(items[0], 64)
		if er != nil {
			log.Fatalf("failed to parse time for filename %s, %s", filename, er)
		}
		times = append(times, t)

		s, er := strconv.ParseInt(items[1], 10, 64)
		if er != nil {
			log.Fatalf("failed to parse size for filename %s, %s", filename, er)
		}
		sizes = append(sizes, int(s))
	}

	features, err := extract(times, sizes)
	if err != nil {
		log.Fatalf("failed to extract features for filename %s, %s", filename, err)
	}
	err = ioutil.WriteFile(new_path + path.Base(filename) + suffix, []byte(features+FeatureDelimiter), 0666)
	if err != nil {
		log.Fatalf("failed to write features file for filename %s, %s", filename, err)
	}
}

func main() {
	folder := flag.String("folder", "../data/cells/", "folder with cell traces")
	sites := flag.Int("sites", 0, "number of sites")
	open := flag.Int("open", 0, "number of open-world sites")
	instances := flag.Int("instances", 0, "number of instances")
	suffix := flag.String("suffix", ".cellf", "the suffix for the resulting files with parsed features")
	new_path := flag.String("new_path", "../data/knn/", "The the path to the folder where the features will be stored")
	flag.Parse()

	// workers
	wg := new(sync.WaitGroup)
	work := make(chan string)
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for filename := range work {
				parse(filename, *suffix, *new_path)
			}
		}()
	}

	log.Printf("starting parsing...")
	// closed world with specified number of instances
	for site := 0; site < *sites; site++ {
		for instance := 0; instance < *instances; instance++ {
			work <- path.Join(*folder, strconv.Itoa(site)+"-"+strconv.Itoa(instance))
		}
	}
	// open world, only one instance per site
	for site := 0; site < *open; site++ {
		work <- path.Join(*folder, strconv.Itoa(site))
	}

	close(work)
	wg.Wait()

	log.Printf("done parsing (%d sites, %d instances, %d open world, folder \"%s\", suffix \"%s\")",
		*sites, *instances, *open, *folder, *suffix)
}
