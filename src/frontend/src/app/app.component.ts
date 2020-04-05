import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpService } from './http.service';
import { CommonService } from './common.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
})
export class AppComponent implements OnInit {

  actions = ["F", "B", "T", "D", "L", "R", "f", "b", "t", "d", "l", "r"];

  constructor(private httpService: HttpService, public commonService: CommonService) { }

  ngOnInit() {
    this.commonService.getSolved();
  }
  // TODO: Før oh med over alt
}
